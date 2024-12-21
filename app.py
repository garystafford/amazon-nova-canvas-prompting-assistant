# Description: Amazon Nova Canvas Prompting Assistant
# Author: Gary A. Stafford
# Date: 2024-12-16
# Amazon Nova Canvas References:
# https://docs.aws.amazon.com/nova/latest/userguide/image-gen-code-examples.html
# https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html

import base64
import datetime
import io
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, DefaultDict, List, Optional, Tuple

import boto3
import streamlit as st
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
REGION_NAME: str = "us-east-1"
MODEL_ID: str = "amazon.nova-canvas-v1:0"
IMAGE_DIRECTORY: str = "./generated_images"
MAX_IMAGE_SIZE: int = 4_096
MIN_IMAGE_SIZE: int = 320
MAX_ASPECT_RATIO: int = 4
MAX_PIXEL_COUNT: int = 4_194_304
MIN_PROMPT_LENGTH: int = 1
MAX_PROMPT_LENGTH: int = 1_024


class NovaCanvasError(Exception):
    """Base exception for all Nova Canvas related errors."""

    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
        logger.error(f"Error {error_code}: {message}" if error_code else message)


class ImageError(NovaCanvasError):
    """Specific exception for image processing errors."""

    pass


class PromptError(Exception):
    """Base exception for prompt-related errors."""

    pass


class PromptValidationError(PromptError):
    """Raised when prompt validation fails."""

    pass


@dataclass(frozen=True)
class Prompt:
    """
    Handles prompt generation and validation for Amazon Nova Canvas.

    Attributes:
        title: Name of the prompt template
        subject: Main subject of the image
        environment: Background/setting
        action: Subject positioning
        lighting: Lighting conditions
        camera: Camera angle and framing
        style: Visual style
    """

    # Instance attributes with defaults
    title: str = field(default="Custom prompt")
    subject: str = field(default="")
    environment: Optional[str] = field(default="")
    action: Optional[str] = field(default="")
    lighting: Optional[str] = field(default="")
    camera: Optional[str] = field(default="")
    style: Optional[str] = field(default="")

    # Class constants
    NEGATION_WORDS: ClassVar[frozenset] = frozenset(
        [
            "no",
            "not",
            "neither",
            "never",
            "no one",
            "nobody",
            "none",
            "nor",
            "nothing",
            "nowhere",
            "without",
            "barely",
            "hardly",
            "scarcely",
            "seldom",
        ]
    )

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate all input fields are strings."""
        for field_name, value in self.__dict__.items():
            if value and not isinstance(value, str):
                raise PromptValidationError(f"{field_name} must be a string")

    def _format_field(self, label: str, value: Optional[str]) -> Optional[str]:
        """Format a field with its label if value exists."""
        return f"{label}: {value.strip()}" if value else None

    def generate_prompt(self) -> str:
        """
        Generate formatted prompt from attributes.

        Returns:
            str: Formatted prompt string

        Raises:
            PromptValidationError: If prompt exceeds max length
        """
        fields = [
            self._format_field("Subject", self.subject),
            self._format_field("Environment", self.environment),
            self._format_field("Subject action, position, and pose", self.action),
            self._format_field("Lighting", self.lighting),
            self._format_field("Camera position and framing", self.camera),
            self._format_field("Image style", self.style),
        ]

        prompt = ", \n".join(filter(None, fields))

        if len(prompt) < MIN_PROMPT_LENGTH:
            raise PromptValidationError(
                f"Prompt must be at least {MIN_PROMPT_LENGTH} character in length."
            )

        if len(prompt) > MAX_PROMPT_LENGTH:
            raise PromptValidationError(
                f"Prompt length {len(prompt)} exceeds maximum {MAX_PROMPT_LENGTH} characters."
            )

        if len(self.subject) < MIN_PROMPT_LENGTH:
            raise PromptValidationError(f"The 'Subject' is a required field.")
        return prompt


@dataclass
class NegativePrompt:
    """Handles negative prompts for Amazon Nova Canvas."""

    title: str
    text: str


@dataclass
class RequestParameters:
    """Handles request parameters for Amazon Nova Canvas."""

    def __init__(
        self,
        prompt="default prompt",
        negative_text="default prompt",
        number_of_images=3,
        quality="premium",
        height=1024,
        width=1024,
        cfg_scale=3.0,
        seed=0,
    ):
        self.prompt: str = prompt
        self.negative_text: str = negative_text
        self.number_of_images: int = number_of_images
        self.quality: str = quality
        self.height: int = height
        self.width: int = width
        self.cfg_scale: float = cfg_scale
        self.seed: int = seed


@dataclass
class ImageDimensions:
    """Handles image dimension validation."""

    def __init__(self, width, height):
        self.width: int = width
        self.height: int = height

    def check_dimensions(self) -> str:
        if self.width > MAX_IMAGE_SIZE or self.height > MAX_IMAGE_SIZE:
            return "Width and height must be less than or equal to 4096 pixels. Please try again."

        if self.width < MIN_IMAGE_SIZE or self.height < MIN_IMAGE_SIZE:
            return "Width and height must be greater than or equal to 320 pixels. Please try again."

        if self.width * self.height > MAX_PIXEL_COUNT:
            return (
                "The total pixel count must be less than 4,194,304. Please try again."
            )
        if (
            self.width / self.height > MAX_ASPECT_RATIO
            or self.height / self.width > MAX_ASPECT_RATIO
        ):
            return (
                "The aspect ratio must be less than or equal to 4:1. Please try again."
            )


@dataclass(frozen=True)
class AspectRatioSelector:
    """Handles image aspect ratio selection and validation."""

    DIMENSIONS: DefaultDict[str, Tuple[int, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: (512, 512),
            {
                "512 x 512 (1:1)": (512, 512),
                "1024 x 1024 (1:1)": (1024, 1024),
                "2048 x 2048 (1:1)": (2048, 2048),
                "1024 x 336 (3:1)": (1024, 336),
                "1024 x 512 (2:1)": (1024, 512),
                "1024 x 576 (16:9)": (1024, 576),
                "1024 x 672 (3:2)": (1024, 672),
                "1024 x 816 (5:4)": (1024, 816),
                "1280 x 720 (16:9)": (1280, 720),
                "2048 x 512 (4:1)": (2048, 512),
                "2288 x 1824 (5:4)": (2288, 1824),
                "2512 x 1664 (3:2)": (2512, 1664),
                "2720 x 1520 (16:9)": (2720, 1520),
                "2896 x 1440 (2:1)": (2896, 1440),
                "3536 x 1168 (3:1)": (3536, 1168),
                "4096 x 1024 (4:1)": (4096, 1024),
                "336 x 1024 (1:3)": (336, 1024),
                "512 x 1024 (1:2)": (512, 1024),
                "512 x 2048 (1:4)": (512, 2048),
                "576 x 1024 (9:16)": (576, 1024),
                "672 x 1024 (2:3)": (672, 1024),
                "720 x 1280 (9:16)": (720, 1280),
                "816 x 1024 (4:5)": (816, 1024),
                "1024 x 4096 (1:4)": (1024, 4096),
                "1168 x 3536 (1:3)": (1168, 3536),
                "1440 x 2896 (1:2)": (1440, 2896),
                "1520 x 2720 (9:16)": (1520, 2720),
                "1664 x 2512 (2:3)": (1664, 2512),
                "1824 x 2288 (4:5)": (1824, 2288),
            },
        )
    )

    def aspect_ratio_options(self) -> List[str]:
        """Returns list of available aspect ratio options grouped by type."""

        return [
            "--- Square ---",
            *[k for k, v in self.DIMENSIONS.items() if v[0] == v[1]],
            "--- Landscape ---",
            *[k for k, v in self.DIMENSIONS.items() if v[0] > v[1]],
            "--- Portrait ---",
            *[k for k, v in self.DIMENSIONS.items() if v[0] < v[1]],
        ]

    def get_dimension(self, aspect_ratio_selection: str) -> Tuple[int, int]:
        """
        Returns dimensions for selected aspect ratio.

        Args:
            aspect_ratio_selection: Selected aspect ratio string

        Returns:
            Tuple of (width, height)

        Raises:
            ValueError: If invalid selection
        """

        if aspect_ratio_selection.startswith("---"):
            return (512, 512)  # Default

        if aspect_ratio_selection not in self.DIMENSIONS:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio_selection}")

        return self.DIMENSIONS[aspect_ratio_selection]

    def validate_dimensions(self) -> bool:
        """Validates if dimensions meet Nova Canvas requirements."""
        if not (
            MIN_IMAGE_SIZE <= self.width <= MAX_IMAGE_SIZE
            and MIN_IMAGE_SIZE <= self.height <= MAX_IMAGE_SIZE
        ):
            return False

        aspect_ratio = max(self.width / self.height, self.height / self.width)
        return aspect_ratio <= self.MAX_RATIO


def main() -> None:
    st.set_page_config(page_title="Amazon Nova Canvas Prompting Assistant")

    hide_decoration_bar_style = """
    <style>
        header {visibility: hidden;}
    </style>"""
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    st.markdown("## Amazon Nova Canvas Prompting Assistant")

    st.markdown(
        """According to [AWS](https://docs.aws.amazon.com/nova/latest/userguide/prompting-image-generation.html), prompting for image generation models differs from prompting for large language models (LLMs). 
                Image generation models do not have the ability to reason or interpret explicit commands. 
                Therefore, it's best to phrase your prompt as if it were an image caption rather than a command or conversation. 
                You might want to include details about the subject, action, environment, lighting, style, and camera position. 
                This tool helps you be mindful of the Amazon Nova Canvas' requirements and best practices when writing a prompt."""
    )

    createForm()


def createForm() -> None:
    """
    Creates a form for generating images using Amazon Nova Canvas.
    The form includes sections for:
    - Image Generation Details: Allows users to select prompt samples and enter custom prompt parameters.
    - Negative Prompt Samples: Allows users to select negative prompt samples or enter custom negative prompts.
    - Image Configuration: Allows users to select image size, quality, prompt strength, number of images, and seed.
    The form also includes a button to generate images based on the provided parameters.
    Returns:
        None
    """

    st.sidebar.markdown("### Image Generation Details")

    st.markdown("### Prompt Parameters")

    prompt_samples = get_prompt_samples()

    prompt_sample_select = st.selectbox(
        label="Prompt samples",
        options=range(len(prompt_samples)),
        index=0,
        format_func=lambda x: prompt_samples[x].title,
        help="Select a prompt sample to use as a starting point for your image or enter your own custom prompt.",
    )

    subject = st.text_input(
        "Subject",
        prompt_samples[prompt_sample_select].subject,
        help="The main subject of the image.",
    )
    environment = st.text_input(
        "Environment (optional)",
        prompt_samples[prompt_sample_select].environment,
        help="The setting or background of the image.",
    )
    action = st.text_input(
        "Subject action, position, and pose (optional)",
        prompt_samples[prompt_sample_select].action,
        help="The action, position, and/or pose of the subject in the image.",
    )
    lighting = st.text_input(
        "Lighting (optional)",
        prompt_samples[prompt_sample_select].lighting,
        help="The lighting conditions of the image.",
    )
    camera = st.text_input(
        "Camera position and framing (optional)",
        prompt_samples[prompt_sample_select].camera,
        help="The camera position and/or framing of the image.",
    )
    style = st.text_input(
        "Image style (optional)",
        prompt_samples[prompt_sample_select].style,
        help="The style and medium of the image (e.g., 'photo', 'illustration', 'painting').",
    )

    st.markdown("---")
    st.markdown("### Negative Prompt")

    negative_text_samples = get_negative_text_samples()

    negative_text_sample_select = st.selectbox(
        label="Negative prompt samples",
        options=range(len(negative_text_samples)),
        index=0,
        format_func=lambda x: negative_text_samples[x].title,
        help="Select a negative prompt sample or enter your own custom negative prompt.",
    )

    negative_text = st.text_area(
        label="Negative prompt",
        value=negative_text_samples[negative_text_sample_select].text,
        height=150,
        help="Use negative prompts to exclude objects or style characteristics that might otherwise naturally occur as a result of your main prompt.",
    )

    st.markdown("---")
    st.markdown("### Image Configuration")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        image_size = st.radio(
            label="Image size",
            options=["Pre-defined", "Custom"],
            index=0,
            horizontal=True,
            help="Select a pre-defined image size and aspect ratio or enter a custom size.",
        )

        selector = AspectRatioSelector()
        if image_size == "Pre-defined":
            aspect_ratio = st.selectbox(
                "Size (px) / Aspect ratio",
                selector.aspect_ratio_options(),
                index=2,
                help="Select a pre-defined image size and aspect ratio.",
            )
            width, height = selector.get_dimension(aspect_ratio)
        else:
            st.markdown("#### Custom size")
            width = st.slider(
                "Width",
                min_value=320,
                max_value=4096,
                value=1024,
                step=16,
                help="The width of the image in pixels.",
            )
            height = st.slider(
                "Height",
                min_value=320,
                max_value=4096,
                value=1024,
                step=16,
                help="The height of the image in pixels.",
            )

        image_dimensions = ImageDimensions(width, height)

        if image_dimensions.check_dimensions():
            st.sidebar.error(image_dimensions.check_dimensions())
            # st.stop

        st.markdown("<br>", unsafe_allow_html=True)
        quality = st.radio(
            label="Quality",
            options=["standard", "premium"],
            index=1,
            horizontal=True,
            help="The quality of the image.",
        )

    with col2:
        cfg_scale = st.slider(
            "Prompt Strength (CFG scale)",
            min_value=1.1,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="How strongly the generated image should adhere to the prompt. Use a lower value to introduce more randomness in the generation.",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        number_of_images = st.slider(
            "Number of images",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="The number of images to generate.",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        seed = st.slider(
            "Seed",
            min_value=0,
            max_value=858_993_459,
            value=42,
            step=1,
            help="The seed for the image generation. Using a different seed, the model is able to generate different images each time, even if all other values stay the same.",
        )
    st.markdown("---")
    submit_button = st.button(label="Generate Image(s)")

    prompt = None
    if submit_button:
        try:
            prompt_template = Prompt(
                title="Custom prompt",
                subject=subject,
                action=action,
                environment=environment,
                lighting=lighting,
                camera=camera,
                style=style,
            )
            prompt = prompt_template.generate_prompt()
            logger.info(f"Prompt: {prompt}")

            if any(word in prompt for word in prompt_template.NEGATION_WORDS):
                words = ", ".join(list(prompt_template.NEGATION_WORDS))
                warnings_message = f"The model doesn't understand negation in a prompt and attempting to use negation will result in the opposite of what you intend. For example, a prompt such as 'a fruit basket with no bananas' will actually signal the model to include bananas in the image. You should removes these words in the prompt: {words}"
                st.sidebar.warning(warnings_message)
                logger.warning(warnings_message)
        except PromptValidationError as err:
            st.sidebar.error(err)
            logger.error(err)
            st.stop()
        except Exception as err:
            st.sidebar.error(f"An error occurred: {err}")
            logger.error(err)
            st.stop()

        display_prompt_details(
            prompt,
            negative_text,
            width,
            height,
            cfg_scale,
            number_of_images,
            quality,
            seed,
        )

        images = []
        with st.spinner("Generating image(s)..."):
            try:
                request_params = RequestParameters(
                    prompt,
                    negative_text,
                    number_of_images,
                    quality,
                    height,
                    width,
                    cfg_scale,
                    seed,
                )

                body = generate_body(request_params)
                start_time = datetime.datetime.now()
                images = generate_image(model_id=MODEL_ID, body=body)
                end_time = datetime.datetime.now()
                total_time = (end_time - start_time).total_seconds()
                time_per_image = total_time / len(images)
                st.sidebar.info(
                    f"Total time to generate {len(images)} images: {total_time:.2f} seconds, or an average of {time_per_image:.2f} seconds per image."
                )
                display_images(images)
                save_images(images)
            except ClientError as err:
                message = err.response["Error"]["Message"]
                logger.error("A client error occurred:", message)
                st.sidebar.error(message)
            except ImageError as err:
                logger.error(err.message)
                st.sidebar.error(err.message)
            else:
                logger.info(
                    f"Finished generating image with Amazon Nova Canvas, model ID: {MODEL_ID}."
                )
    st.markdown(
        "<small style='color: #ACADC1'> Gary A. Stafford, 2024</small>",
        unsafe_allow_html=True,
    )


def get_negative_text_samples() -> list[dict]:
    """
    Retrieve a list of dictionaries containing negative text samples.
    Each dictionary in the list represents a category of negative text samples with a title and corresponding text.
    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - title (str): The category title of the negative text sample.
            - text (str): A string of negative text samples related to the category.
    """

    negative_text_samples = [
        NegativePrompt(
            title="Custom (blank)",
            text=" ",
        ),
        NegativePrompt(
            title="Avoid poor general image quality",
            text="blurry, blur, censored, censored, crop, cut off, draft, draft, grainy, good, out of focus, out of frame, out of focus, out of frame, poorly lit, poor quality, poorly lit, shot, shadow, worst quality, worst quality",
        ),
        NegativePrompt(
            title="Avoid text in the image",
            text="annotations, artist name, autograph, caption, digits, error, initials, inscription, label, letters, logo, name, seal, signature, signature, stamp, textual elements, trademark, typography, username, watermark, words, writing",
        ),
        NegativePrompt(
            title="Avoid distorted human features",
            text="bad anatomy, bad body, bad eyes, bad face, bad hands, bad arms, bad legs, bad teeth, deformities, extra fingers, extra limbs, extra toes, missing limbs, missing fingers, mutated, malformed, mutilated, morbid, 3d character",
        ),
        NegativePrompt(
            title="Avoid poor photo quality",
            text="anime, asymmetrical, bad art, bad photography, bad photo, black and white, blur, blurry, blend, blue facial, cartoon, censored, CGI, copy, cut off, draft, duplicate, digital, double exposure, facial, final, grainy, grayscale, good, graphics, graphic novel, graphics, glue, glitch, gradient, low details, low-res, low quality, manga, merge, out of frame, over-saturated, overexposed, poor photo, poor photography, poor quality, render, shadow, shot",
        ),
    ]

    return negative_text_samples


def get_prompt_samples() -> list[Prompt]:
    """
    Generates a list of predefined prompt samples for various artistic and photographic styles.
    Returns:
        list: A list of Prompt objects, each containing the following attributes:
            - title (str): The title or category of the prompt.
            - subject (str): The main subject of the prompt.
            - environment (str): The setting or background environment of the prompt.
            - action (str): The action, position, and/or pose of the subject within the environment.
            - lighting (str): The type of lighting used in the scene.
            - camera (str): The camera angle and other photographic details.
            - style (str): The artistic or photographic style of the prompt.
    """

    prompt_samples = [
        Prompt(
            title="Custom prompt (blank)",
            subject="",
            environment="",
            action="",
            lighting="",
            camera="",
            style="",
        ),
        Prompt(
            title="Oil painting of a cat",
            subject="Calico colored cat",
            environment="Cozy living room",
            action="Lounging on a sofa",
            lighting="Soft lighting",
            camera="High angle",
            style="Oil on canvas",
        ),
        Prompt(
            title="Stock photo of a teacher",
            subject="Female teacher with a warm smile",
            environment="Grade-school classroom with blackboard in background",
            action="Standing in front of a blackboard",
            lighting="Clean white light",
            camera="Eye-level facing the teacher, shallow depth of field, blurred background",
            style="Realistic editorial photo, stock photography, high-quality",
        ),
        Prompt(
            title="Illustration of a woman on a ship",
            subject="Woman in a large hat",
            environment="Boat deck with a railing and ocean view",
            action="Standing at the ship's railing, looking out across the ocean, left side of the frame",
            lighting="Golden hour light, setting sun",
            camera="Eye-level from behind the woman, looking out across the ocean",
            style="Ethereal soft-shaded, story, illustration",
        ),
        Prompt(
            title="Fashion photography of a male model",
            subject="Cool looking stylish man in an orange jacket, dark skin, wearing reflective glasses",
            environment="Aqua blue sleek building shapes in background",
            action="Standing in front of the building",
            lighting="Natural light",
            camera="Slightly low angle, face and chest in view",
            style="High-quality fashion photography, editorial, modern, sleek, sharp",
        ),
        Prompt(
            title="Dragon illustration",
            subject="Large, menacing dragon",
            environment="Medieval castle ruins",
            action="Roaring and breathing fire",
            lighting="Dark and moody",
            camera="Low angle, wide shot",
            style="Fantasy, epic, dark, detailed, illustration",
        ),
        Prompt(
            title="The Batmobile in LA traffic",
            subject="The Batmobile from the Batman movies",
            environment="Los Angeles traffic, ",
            action="Stuck in Los Angeles traffic",
            lighting="rainy, wet, reflections",
            camera="wide shot",
            style="Impressionist painting, large broad brush strokes, impasto",
        ),
        Prompt(
            title="Western cowboy tintype",
            subject="Western cowboy, Caucasian, White, male, man",
            environment="Saloon background, 1880's, Old West",
            action="Looking off directly at camera",
            lighting="Soft, warm, tungsten, candlelight",
            camera="portrait, head and shoulder, narrow depth of field, blurred background",
            style="Tintype, vintage photograph, sepia tone, hand-made print, imperfections",
        ),
        Prompt(
            title="Japanese-style woodblock print",
            subject="Brown-eared Bulbul perched on a cherry blossom branch",
            environment="Springtime cherry blossom tree",
            action="Bird and branch on right side of image, bird facing left",
            lighting="",
            camera="",
            style="Japanese-style, Katsushika Hokusai, woodblock print, ukiyo-e print, Edo period, handmade Washi paper",
        ),
        Prompt(
            title="Oil painting of Tango dancers",
            subject="Man and woman both dressed in formal attire with the woman wearing a red dress",
            environment="Dark gray, creating an atmosphere of mystery and elegance",
            action="Dancing the tango",
            lighting="",
            camera="",
            style="Oil painting",
        ),
    ]

    return prompt_samples


def display_images(images) -> None:
    """
    Display a list of images using Streamlit.
    Parameters:
    images (list): A list of image objects or file paths to be displayed.
    Returns:
    None
    """

    st.image(images, use_container_width=True)


def save_images(images) -> None:
    """
    Save a list of images to the local filesystem.

    Args:
        images (list): A list of image bytes to be saved.

    The images are saved in the './generated_images/' directory with a unique filename
    for each image in JPEG format.
    """
    for image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"{IMAGE_DIRECTORY}/image-{str(uuid.uuid4())}.jpg", "JPEG")


def display_prompt_details(
    prompt, negative_text, width, height, cfg_scale, number_of_images, quality, seed
) -> None:
    """
    Display the details of the prompt configuration and results in the Streamlit sidebar.
    Parameters:
    prompt (str): The main text prompt.
    negative_text (str): The negative prompt text.
    width (int): The width of the generated images.
    height (int): The height of the generated images.
    cfg_scale (float): The CFG scale value.
    number_of_images (int): The number of images to generate.
    quality (str): The quality setting for the images.
    seed (int): The seed value for random number generation.
    Returns:
    None
    """

    with st.sidebar:
        st.markdown(f"- Prompt: {prompt}")
        st.markdown(f"- Negative prompt: {negative_text}")
        st.markdown(f"- Width: {width}")
        st.markdown(f"- Height: {height}")
        st.markdown(f"- Quality: {quality}")
        st.markdown(f"- CFG scale: {cfg_scale}")
        st.markdown(f"- Seed: {seed}")
        st.markdown(f"- Number of images: {number_of_images}")
        st.markdown("### Results")


def generate_body(request_params: RequestParameters) -> str:
    """
    Entrypoint for Amazon Nova Canvas  example.
    """

    request_params = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": request_params.prompt,
                "negativeText": request_params.negative_text,
            },
            ""
            "imageGenerationConfig": {
                "numberOfImages": request_params.number_of_images,
                "quality": request_params.quality,
                "height": request_params.height,
                "width": request_params.width,
                "cfgScale": request_params.cfg_scale,
                "seed": request_params.seed,
            },
        }
    )

    logger.info("Request body: %s", request_params)
    return request_params


def generate_image(model_id, body) -> list[bytes]:
    """
    Generate an image using Amazon Nova Canvas model on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        image_bytes (bytes): The image generated by the model.
    """

    bedrock = get_bedrock_client()

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    images = []
    for image in response_body.get("images"):
        base64_bytes = image.encode("ascii")
        image_bytes = base64.b64decode(base64_bytes)
        images.append(image_bytes)

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise ImageError(f"Image generation error. Error is {finish_reason}")

    logger.info(
        "Successfully generated %s image(s) with Amazon Nova Canvas, model ID: %s",
        str(len(images)),
        model_id,
    )

    st.sidebar.success(
        f"Successfully generated {len(images)} image with Amazon Nova Canvas, model ID: {model_id}."
    )

    return images


def get_bedrock_client() -> boto3.client:
    """
    Creates and returns a boto3 client for the Bedrock runtime service.

    This function initializes a boto3 client configured to interact with the
    Bedrock runtime service. The client is configured with a read timeout of
    300 seconds and uses the region specified by the REGION_NAME constant.

    Returns:
        botocore.client.BedrockRuntime: A boto3 client for the Bedrock runtime service.
    """
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        config=Config(
            read_timeout=300,
            region_name=REGION_NAME,
        ),
    )
    return bedrock_client


if __name__ == "__main__":
    main()

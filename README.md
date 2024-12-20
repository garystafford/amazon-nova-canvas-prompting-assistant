# Amazon Nova Canvas Prompting Assistant

Prompting for image generation models differs from prompting for large language models (LLMs).
Image generation models do not have the ability to reason or interpret explicit commands.
Therefore, it's best to phrase your prompt as if it were an image caption rather than a command or conversation.
You might want to include details about the subject, action, environment, lighting, style, and camera position.

![Image Examples](./ui_preview/canvas_examples.png)

This Streamlit-based UI helps you be mindful of Amazon Nova Canvas' requirements and best practices. Features of UI include:

### Prompt Parameters

- Prompt Template: Breaks down prompt input into separate aspects: subject, action, environment, lighting, style, and camera position
- Prompt Samples: Provides sample prompt as a reference
- Prompt Negation: Checks for the use negation words in the prompt and issues a warning
- Prompt Length: Confirms the prompt is within the maximum character length
- Negative Prompt Samples: Provides sample negative prompt as a reference

### Image Configuration

- Image Size and Aspect Ratio: Easy-to-use controls for pre-selected image sizes and aspect ratios, or custom sizes
- Image Size: Confirms the image is within the maximum total pixel size limit
- Image Aspect Ratio: Confirms the image is within the maximum aspect ratio limit
- Image Dimensions: Confirms the image is within the minimum and maximum pixel width and height limits
- Image Display and Saving: Both displays the generated image and saves them to a local directory

## UI Preview

![UI Preview](./ui_preview/canvas_ui_01.png)

![UI Preview](./ui_preview/canvas_ui_02.png)

![UI Preview](./ui_preview/canvas_ui_03.png)

## Amazon Nova References

- [User Guide for Amazon Nova: Amazon Nova Canvas prompting best practices](https://docs.aws.amazon.com/nova/latest/userguide/prompting-image-generation.html)
- [User Guide for Amazon Nova: Sample Code](https://docs.aws.amazon.com/nova/latest/userguide/image-gen-code-examples.html)
- [User Guide for Amazon Nova: Request and response structure for image generation](https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html)

## Prerequisites

The only prerequisites are a recently version of Python and AWS IAM console access to Amazon Bedrock and Amazon Nova Canvas model.

## Mac: Configure Environment

Make sure you have provided your AWS credential on the commandline, or using an alternative authentication method, before starting the application.

```sh
export AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
export AWS_SESSION_TOKEN="<YOUR_AWS_SESSION_TOKEN>"
```

Create a virtual Python environment

```sh
python --version # I am using Python 3.13.0

python -m pip install virtualenv -U # --break-system-packages
python -m venv .venv
source .venv/bin/activate
```

Install Python package dependencies

```sh
python -m pip install pip -U
python -m pip install -r requirements.txt -U

streamlit --version
```

Deactivate and delete virtual environment once you are done

```sh
deactivate
rm -rf .venv
```

## Windows: Configure Environment

Make sure you have provided your AWS credential on the commandline, or using an alternative authentication method, before starting the application.

```bat
set AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
set AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
set AWS_SESSION_TOKEN="<YOUR_AWS_SESSION_TOKEN>"
```

Create a virtual Python environment

```bat
python -m venv .venv
.venv\Scripts\activate

where python
```

Install Python package dependencies

```bat
python -m pip install pip -U
python -m pip install -r requirements.txt -U
```

Deactivate and delete virtual environment once you are done

```bat
deactivate
rm -rf .venv
```

## Mac: Run the Streamlit application

```sh
streamlit run app.py \
    --server.runOnSave true \
    --theme.base "dark" \
    --theme.backgroundColor "#26273B" \
    --theme.primaryColor "#ACADC1" \
    --theme.secondaryBackgroundColor "#454560" \
    --theme.font "sans serif"\
    --ui.hideTopBar "true" \
    --client.toolbarMode "minimal"
```

## Windows: Run the Streamlit application

```bat
streamlit run app.py ^
    --server.runOnSave true ^
    --theme.base "dark" ^
    --theme.backgroundColor "#26273B" ^
    --theme.primaryColor "#ACADC1" ^
    --theme.secondaryBackgroundColor "#454560" ^
    --theme.font "sans serif"^
    --ui.hideTopBar "true" ^
    --client.toolbarMode "minimal"
```

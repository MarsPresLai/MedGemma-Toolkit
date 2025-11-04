# MedGemma API

This is an API built with FastAPI to run the MedGemma model locally. It provides endpoints for both Image Visual Question Answering (VQA) and text-only chat.

## Features

- **Image VQA**: Upload an image and ask a question to get and answer.
- **Text-only Chat**: Have a conversation with the MedGemma model.
- **FastAPI**: Powered by FastAPI, with auto-generated interactive documentation.
- **4-bit Quantization**: Uses `bitsandbytes` for 4-bit quantization to reduce VRAM usage.

## Getting Started

Please follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Navigate to the API directory:
   ```bash
   cd medgemma-api
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

To access Hugging Face models that require authentication, you need to set the `HF_TOKEN` environment variable.

```bash
export HF_TOKEN=your_hugging_face_token
```

### Running the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Usage

You can interact with the API using `curl` or any HTTP client.

### `/predict/text`

This endpoint is for text-only chat.

- **URL**: `/predict/text`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "prompt": "What are the symptoms of a common cold?",
    "system": "You are a helpful medical assistant."
  }
  ```
- **Curl Example**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/predict/text' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "prompt": "What are the symptoms of a common cold?",
    "system": "You are a helpful medical assistant."
  }'
  ```

### `/predict/image`

This endpoint is for Image Visual Question Answering.

- **URL**: `/predict/image`
- **Method**: `POST`
- **Request Body**: `multipart/form-data`
  - `prompt` (string): The user's question.
  - `image` (file): The image file.
- **Curl Example**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/predict/image' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'prompt=What are the key findings in this picture?' \
    -F 'image=@/path/to/your/image.jpg;type=image/jpeg'
  ```

## Project Structure

```
.
├── main.py           # The FastAPI application
├── requirements.txt  # Python dependencies
└── README.md         # This document
```

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - For the model and processor
- [PyTorch](https://pytorch.org/) - The machine learning framework
- [Uvicorn](https://www.uvicorn.org/) - The ASGI server
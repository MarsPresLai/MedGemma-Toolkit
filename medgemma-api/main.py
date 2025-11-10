from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
import os
from huggingface_hub import login
import uvicorn

# Import the client for the second LLM
from llm_client import query_medical_llm

app = FastAPI()
processor = None
model = None

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins like ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static files ---
# Serve the new React frontend from the built dist folder
# Make sure to build the frontend first: cd ../medgemma-frontend && npm run build
app.mount("/assets", StaticFiles(directory="../medgemma-frontend/dist/assets"), name="assets")

# Serve favicon and other static files from dist root
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("../medgemma-frontend/dist/favicon.ico")

class TextRequest(BaseModel):
    prompt: str
    system: str | None = "You are a helpful medical assistant."

# --- Model Loading (at startup) ---
@app.on_event("startup")
async def startup_event():
    global processor, model

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[Info] Found HF_TOKEN, logging in...")
        login(token=hf_token)
    else:
        print("[Warn] HF_TOKEN not set.")

    dtype = torch.bfloat16
    device_map = "auto"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("[Info] Using 4-bit quantization.")

    model_id = "google/medgemma-4b-it"
    print(f"[Info] Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    print(f"[Info] Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )

# --- Refactored Prediction Logic ---

def check_model_loaded():
    if not processor or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few seconds.")

async def run_text_prediction(request: TextRequest) -> str:
    """Runs text-based prediction with MedGemma."""
    check_model_loaded()
    system_content = [{"type": "text", "text": request.system}]
    user_content = [{"type": "text", "text": request.prompt}]
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    gen = out[0][input_len:]
    return processor.decode(gen, skip_special_tokens=True)

async def run_image_prediction(prompt: str, image: UploadFile) -> str:
    """Runs image-based prediction with MedGemma."""
    check_model_loaded()
    image_pil = Image.open(image.file).convert("RGB")
    content = [{"type": "image"}, {"type": "text", "text": prompt}]
    
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=[image_pil], text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    gen = out[0][input_len:]
    return processor.decode(gen, skip_special_tokens=True)

def extract_answer(response: str) -> str:
    try:
        start_tag = "<Answer>"
        end_tag = "</Answer>"
        start_index = response.index(start_tag) + len(start_tag)
        end_index = response.index(end_tag, start_index)
        return response[start_index:end_index].strip()
    except ValueError:
        return response

# --- API Endpoints ---

@app.get("/")
def read_root():
    return FileResponse("../medgemma-frontend/dist/index.html")

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    decoded = await run_text_prediction(request)
    return {"response": decoded}

@app.post("/predict/image")
async def predict_image(prompt: str = Form(...), image: UploadFile = File(...)):
    decoded = await run_image_prediction(prompt, image)
    return {"response": decoded}

@app.post("/predict/chained_text")
async def predict_chained_text(request: TextRequest):
    # Step 1: Get initial analysis from MedGemma
    medgemma_response = await run_text_prediction(request)
    
    # Step 2: Use MedGemma's response as a prompt for the second LLM
    final_response = query_medical_llm(medgemma_response)
    
    return {
        "medgemma_response": medgemma_response,
        "final_response": final_response
    }

@app.post("/predict/chained_image")
async def predict_chained_image(prompt: str = Form(...), image: UploadFile = File(...)):
    # Step 1: Get initial analysis from MedGemma
    medgemma_response = await run_image_prediction(prompt, image)
    
    # Step 2: Use MedGemma's response as a prompt for the second LLM
    final_response = query_medical_llm(medgemma_response)
    final_response = extract_answer(final_response)
    return {
        "medgemma_response": medgemma_response,
        "final_response": final_response
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
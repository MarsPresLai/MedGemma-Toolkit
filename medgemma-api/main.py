
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()

# --- CORS Middleware ---
# This is to allow the frontend to make requests to the API
# For production, you should restrict the origins to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["""*"""],  # Allows all origins
    allow_credentials=True,
    allow_methods=["""*"""],  # Allows all methods
    allow_headers=["""*"""],  # Allows all headers
)

# --- Global variables ---
processor = None
model = None

class TextRequest(BaseModel):
    prompt: str
    system: str | None = "You are a helpful medical assistant."

@app.on_event("startup")
async def startup_event():
    global processor, model

    # --- Login via HF_TOKEN if present ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[Info] Found HF_TOKEN in env, logging in…")
        login(token=hf_token)
    else:
        print("[Warn] HF_TOKEN not set. If the repo requires auth, set it via `export HF_TOKEN=...`")

    # --- Device & dtype ---
    dtype = torch.bfloat16
    device_map = "auto"

    # --- 4bit quantization config (optional) ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("[Info] Using 4-bit quantization (bitsandbytes).")

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the MedGemma API"}

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    if not processor or not model:
        return {"error": "Model not loaded yet. Please try again in a few seconds."}

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
    decoded = processor.decode(gen, skip_special_tokens=True)

    return {"response": decoded}

@app.post("/predict/image")
async def predict_image(prompt: str = Form(...), image: UploadFile = File(...)):
    if not processor or not model:
        return {"error": "Model not loaded yet. Please try again in a few seconds."}

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
    decoded = processor.decode(gen, skip_special_tokens=True)

    return {"response": decoded}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

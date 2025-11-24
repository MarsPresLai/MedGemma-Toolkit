from __future__ import annotations

import base64
import json
import logging
import re
from typing import AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    ollama_base_url: str = Field(default="http://ollama:11434")
    vlm_model: str = Field(default="hf.co/unsloth/medgemma-27b-it-GGUF:Q4_K_M")
    llm_model: str = Field(default="llama3.1:latest")
    request_timeout_seconds: float = Field(default=180.0, gt=0)


settings = Settings()

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Medical Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_vlm_prompt(prompt: str, has_image: bool) -> str:
    if has_image:
        system_prompt = "You are a medical specialist."
        user_prompt = prompt or "What is the key information in the provided medical image?"
    else:
        system_prompt = "You are a helpful medical assistant."
        user_prompt = prompt

    return (
        f"System: {system_prompt}\n"
        f"User: {user_prompt}\n"
        "Assistant:"
    )


def _extract_answer_only(text: str) -> str:
    """
    Extracts content from the last <Answer>...</Answer> tag in the text.

    This function searches for all occurrences of content wrapped in <Answer> tags,
    and returns the content of the last one found. This is useful when a model
    output contains intermediate thoughts and a final answer, ensuring only the
    final answer is returned.

    Args:
        text: The input string, potentially containing <Answer> tags.

    Returns:
        The content of the last found <Answer> tag, stripped of whitespace.
        If no tags are found, the original text is returned, stripped of whitespace.
    """
    # This pattern finds content inside <Answer>...</Answer> tags.
    # re.DOTALL allows `.` to match newlines for multi-line answers.
    # re.IGNORECASE matches tags regardless of case (e.g., <answer>).
    # The `(.*?)` is a non-greedy match to capture the content.
    pattern = re.compile(r"<\s*Answer\s*>(.*?)<\s*/\s*Answer\s*>", re.IGNORECASE | re.DOTALL)

    matches = pattern.findall(text)

    if matches:
        # If there are one or more matches, return the last one, stripped of whitespace.
        return matches[-1].strip()

    # If no <Answer> tags are found, return the original text stripped of whitespace.
    return text.strip()


async def _ollama_token_stream(
    model: str,
    prompt_text: str,
    image_b64: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Streams tokens from Ollama's generate API.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "options": {"temperature": 0},
    }
    if image_b64:
        payload["images"] = [image_b64]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds)) as client:
            async with client.stream("POST", f"{settings.ollama_base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except httpx.HTTPError as exc:
        logger.exception("Streaming generate request failed.")
        raise RuntimeError(f"Ollama streaming request failed: {exc}") from exc


def _build_report_prompt(prompt: str, vlm_output: str) -> str:
    return (
        "Generate a structured medical report based on the following clinician prompt and vision findings.\n\n"
        f"Clinician prompt: {prompt}\n\n"
        f"Vision findings: {vlm_output}\n\n"
        "The report must contain the following sections in order:\n"
        "1. Summary\n"
        "2. Supporting Evidence\n"
        "3. Differential Diagnosis\n"
        "4. Recommended Next Steps\n\n"
        "Begin the report directly with the 'Summary' section. Do not include any introductory phrases or conversational text.\n"
        "Explicitly state any uncertainties and base the report strictly on the provided vision findings.\n"
    )


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze/")
async def analyze_case(
    prompt: str = Form(..., min_length=0, max_length=2000),
    image: Optional[UploadFile] = File(default=None),
) -> EventSourceResponse:
    image_b64: Optional[str] = None

    if image:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        image_b64 = base64.b64encode(content).decode("utf-8")

    async def event_stream() -> AsyncIterator[str]:
        vlm_buffer = ""
        llm_buffer = ""
        try:
            yield json.dumps({"event": "status", "state": "processing"})

            vlm_prompt = _build_vlm_prompt(prompt, bool(image_b64))
            async for token in _ollama_token_stream(settings.vlm_model, vlm_prompt, image_b64=image_b64):
                vlm_buffer += token
                yield json.dumps({"event": "vlm_token", "token": token})

            yield json.dumps({"event": "vlm_complete", "vlm_output": vlm_buffer})

            report_prompt = _build_report_prompt(prompt, vlm_buffer)
            async for token in _ollama_token_stream(settings.llm_model, report_prompt):
                llm_buffer += token
                yield json.dumps({"event": "llm_token", "token": token})

            llm_report = _extract_answer_only(llm_buffer)
            yield json.dumps({"event": "done", "vlm_output": vlm_buffer, "llm_report": llm_report})
        except Exception as exc:
            logger.exception("Streaming pipeline failed.")
            yield json.dumps({"event": "error", "message": str(exc)})

    return EventSourceResponse(event_stream(), media_type="text/event-stream")

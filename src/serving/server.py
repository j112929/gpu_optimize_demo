"""
FastAPI Server for LLM Serving.
Compatible with OpenAI Chat Completion API.

Integrates with ContinuousBatcher for high-throughput inference.
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse
from typing import AsyncGenerator
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Internal imports
from src.inference.batching import ContinuousBatcher, BatchConfig, Request
from src.serving.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionResponseUsage
)

app = FastAPI(title="GPU Optimize Demo LLM Server")

# Global variables
engine: ContinuousBatcher = None
tokenizer: 'MockTokenizer' = None
model_name: str = "gpu-optimized-model"

class MockTokenizer:
    """Simple tokenizer wrapper."""
    def encode(self, text):
        return [1, 2, 3] # Mock
    def decode(self, ids):
        return " token" # Mock
    @property
    def eos_token_id(self):
        return 2

def create_server(model_path: str, port: int = 8000, tp_size: int = 1):
    """Factory to create and run server."""
    global engine, tokenizer, model_name
    
    model_name = model_path
    
    if model_path == "mock-model":
        print("Loading Mock Model for testing...")
        vocab_size = 32000
        
        class ServerMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 1024)
                self.head = nn.Linear(1024, vocab_size)
            def forward(self, input_ids):
                h = self.emb(input_ids)
                logits = self.head(h)
                from collections import namedtuple
                Output = namedtuple('Output', ['logits'])
                return Output(logits=logits)
                
        model = ServerMockModel()
        tokenizer = MockTokenizer()
    else:
        print(f"Loading Real Model: {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise e
    
    config = BatchConfig(
        max_batch_size=64,
        max_waiting_time_ms=10,
        chunked_prefill_enabled=True,
        max_sequence_length=2048
    )
    
    engine = ContinuousBatcher(model, config, tokenizer)
    engine.start()
    
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.on_event("shutdown")
def shutdown_event():
    if engine:
        engine.stop()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion request."""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    # 1. Tokenize (Simple concatenation for demo)
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role}: {msg.content}\n"
        
    # Use tokenizer to get IDs
    input_ids_list = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long) # [1, Seq]

    # 2. Create Request
    req_id = f"req-{os.urandom(4).hex()}"
    queue = asyncio.Queue()

    # ... callbacks ...
    
    loop = asyncio.get_running_loop()
    
    def on_token(token_id):
        loop.call_soon_threadsafe(queue.put_nowait, token_id)
        
    def on_complete(tokens):
        loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel

    engine_req = Request(
        id=req_id,
        input_ids=input_ids,  # Pass the Tensor!
        prompt=prompt,
        max_new_tokens=request.max_tokens or 256,
        temperature=request.temperature or 1.0,
        on_token=on_token,
        on_complete=on_complete
    )
    
    # Submit
    engine.submit(engine_req)

    # 3. Stream or Return Full
    if request.stream:
        return StreamingResponse(
            stream_generator(req_id, request.model, queue, MockTokenizer()),
            media_type="text/event-stream"
        )
    else:
        # Collect all tokens
        tokens = []
        while True:
            token = await queue.get()
            if token is None:
                break
            tokens.append(token)
            
        full_text = "".join([f" token_{t}" for t in tokens]) # Mock decode
        
        return ChatCompletionResponse(
            id=req_id,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionResponseUsage(
                prompt_tokens=10,
                completion_tokens=len(tokens),
                total_tokens=10 + len(tokens)
            )
        )

async def stream_generator(req_id, model, queue, tokenizer):
    """Generate SSE stream."""
    i = 0
    while True:
        token = await queue.get()
        if token is None:
            # Finish
            resp = ChatCompletionStreamResponse(
                id=req_id,
                model=model,
                choices=[ChatCompletionStreamResponseChoice(
                    index=0, 
                    delta=ChatMessage(role="", content=""), 
                    finish_reason="stop"
                )]
            )
            yield f"data: {resp.json()}\n\n"
            yield "data: [DONE]\n\n"
            break
            
        text = tokenizer.decode([token])
        resp = ChatCompletionStreamResponse(
            id=req_id,
            model=model,
            choices=[ChatCompletionStreamResponseChoice(
                index=0, 
                delta=ChatMessage(role="assistant", content=text), 
                finish_reason=None
            )]
        )
        yield f"data: {resp.json()}\n\n"
        i += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Inference Server")
    parser.add_argument("--model", type=str, default="mock-model", help="Path to model or 'mock-model'")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size (not fully implemented yet)")
    args = parser.parse_args()
    
    create_server(args.model, port=args.port, tp_size=args.tp)

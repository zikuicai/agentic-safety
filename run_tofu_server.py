from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import threading
import sys
import time
import uuid
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 1000
    temperature: float = 0.0


class ModelResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list
    usage: dict
    system_fingerprint: str


class ModelAPI:
    def __init__(self, model_name_or_path, tokenizer_name_or_path, hf_cache_dir=None):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
                                                          cache_dir=hf_cache_dir,
                                                          load_in_4bit=True,  # Enable 4-bit quantization
                                                          device_map='auto',
                                                          bnb_4bit_compute_dtype=torch.float16
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, cache_dir=hf_cache_dir)

        self.app = FastAPI()
        self.server = None
        self.thread = None

        @self.app.post("/v1/chat/completions", response_model=ModelResponse)
        async def chat_completions(request: CompletionRequest):
            if request.model != self.model_name_or_path:
                return JSONResponse(status_code=400,
                                    content={"error": {"message": "Invalid model name provided.", "code": 'null'}})

            messages = request.messages
            print(f'Received messages: {messages}')

            if request.max_tokens < 0:
                request.max_tokens = sys.maxsize
            formatted_messages = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                                    add_generation_prompt=True)

            input_tokens = self.tokenizer(formatted_messages, return_tensors="pt", add_special_tokens=False)
            input_tokens = {key: tensor.to(self.model.device) for key, tensor in input_tokens.items()}

            with torch.cuda.amp.autocast():
                outputs = self.model.generate(**input_tokens, max_length=request.max_tokens,
                                              temperature=request.temperature)
            decoded_output = self.tokenizer.decode(outputs[0][input_tokens['input_ids'].size(1):],
                                                   skip_special_tokens=True)
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": decoded_output
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(input_tokens["input_ids"][0]),
                    "completion_tokens": len(outputs[0]),
                    "total_tokens": len(input_tokens["input_ids"][0]) + len(outputs[0])
                },
                "system_fingerprint": request.model
            }
            print(f'Generated response: {response}')
            return response

    def start(self, port: int):
        config = uvicorn.Config(self.app, host="0.0.0.0", port=port, log_level="info")
        self.server = uvicorn.Server(config)

        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.should_exit = True
            self.thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model API server.")
    parser.add_argument("--model", type=str, default='OPTML-Group/TOFU-origin-Llama-2-7b-chat',
                        help="HF model name or path.")
    parser.add_argument("--tokenizer", type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help="HF tokenizer name or path.")
    parser.add_argument("--hf_cache_dir", type=str, default='.cache/hf',
                        help="Path to the Hugging Face cache directory to be used.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on.")

    args = parser.parse_args()

    api = ModelAPI(model_name_or_path=args.model,
                   tokenizer_name_or_path=args.tokenizer,
                   hf_cache_dir=args.hf_cache_dir)
    api.start(port=args.port)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        api.stop()

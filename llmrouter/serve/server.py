"""
LLMRouter OpenAI-Compatible Server
==================================
Provides an OpenAI-compatible API that integrates directly with OpenClaw and other frontends.

Usage:
    llmrouter serve --config serve_config.yaml

Or via code:
    from llmrouter.serve import create_app, run_server
    app = create_app(config_path="serve_config.yaml")
    run_server(app, port=8000)
"""

import json
import os
import sys
import re
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any, List

# FastAPI
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import httpx
    import uvicorn
except ImportError:
    print("Please install: pip install fastapi uvicorn httpx pydantic")
    sys.exit(1)

from .config import ServeConfig, LLMConfig


# ============================================================
# Request/Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False


# ============================================================
# Router Integration
# ============================================================

class RouterAdapter:
    """LLMRouter adapter"""

    def __init__(self, router_name: str, config_path: Optional[str] = None, fail_on_error: bool = False):
        self.router_name = router_name
        self.config_path = config_path
        self.fail_on_error = fail_on_error
        self.router = None
        self.last_router_info = None
        self._load_router()

    def _load_router(self):
        """Load router"""
        try:
            # Add LLMRouter root directory to path
            llmrouter_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if llmrouter_root not in sys.path:
                sys.path.insert(0, llmrouter_root)

            if self.router_name == "randomrouter":
                from custom_routers.randomrouter.router import RandomRouter
                self.router = RandomRouter(self.config_path)

            elif self.router_name == "thresholdrouter":
                from custom_routers.thresholdrouter.router import ThresholdRouter
                self.router = ThresholdRouter(self.config_path)

            else:
                # Dynamic loading
                import importlib
                module = importlib.import_module(f"custom_routers.{self.router_name}.router")
                for attr in dir(module):
                    if "router" in attr.lower() and not attr.startswith("_"):
                        RouterClass = getattr(module, attr)
                        if hasattr(RouterClass, "route_single"):
                            self.router = RouterClass(self.config_path)
                            break

            print(f"[OK] Router loaded: {self.router_name}")

        except Exception as e:
            print(f"[WARN] Failed to load router '{self.router_name}': {e}")
            print("   Falling back to random selection")
            self.router = None

    def route(self, query: str, available_models: List[str]) -> str:
        """Select model"""
        if self.router is None:
            import random
            return random.choice(available_models)

        try:
            result = self.router.route_single({"query": query})
            model_name = result.get("model_name") or result.get("predicted_llm")
            self.last_router_info = {
                "method": result.get("method"),
                "routing_reason": result.get("routing_reason"),
                "routing_signals": result.get("routing_signals"),
                "routing_confidence": result.get("routing_confidence"),
                "routing_judge_latency_ms": result.get("routing_judge_latency_ms"),
            }

            # Check if model is available
            if model_name in available_models:
                return model_name

            # Fuzzy match
            for m in available_models:
                if model_name and (model_name.lower() in m.lower() or m.lower() in model_name.lower()):
                    return m

            # Fallback
            return available_models[0]

        except Exception as e:
            self.last_router_info = {"error": f"{type(e).__name__}: {e}"}
            print(f"[Router] Error: {type(e).__name__}: {e}")
            if self.fail_on_error:
                raise
            return available_models[0]


# ============================================================
# LLM Backend
# ============================================================

class LLMBackend:
    """LLM backend caller"""

    def __init__(self, config: ServeConfig):
        self.config = config

    async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
                   temperature: Optional[float] = None, stream: bool = False):
        """Call LLM"""
        if llm_name not in self.config.llms:
            raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

        llm_config = self.config.llms[llm_name]
        api_key = llm_config.api_key or self.config.get_api_key(llm_config.provider)

        if stream:
            return self._call_streaming(llm_config, messages, max_tokens, temperature, api_key)
        else:
            return await self._call_sync(llm_config, messages, max_tokens, temperature, api_key)

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: str) -> Dict:
        """Synchronous call"""
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": messages,
                "max_tokens": min(max_tokens, llm.max_tokens),
            }
            if temperature is not None:
                body["temperature"] = temperature

            resp = await client.post(
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

            return resp.json()

    async def _call_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                              temperature: Optional[float], api_key: str) -> AsyncGenerator:
        """Streaming call"""
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": messages,
                "max_tokens": min(max_tokens, llm.max_tokens),
                "stream": True
            }
            if temperature is not None:
                body["temperature"] = temperature

            async with client.stream(
                "POST",
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    yield f'data: {json.dumps({"error": error.decode()[:200]})}\n\n'
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"


# ============================================================
# FastAPI App
# ============================================================

def create_app(config: ServeConfig = None, config_path: str = None) -> FastAPI:
    """Create FastAPI application"""

    if config is None and config_path:
        config = ServeConfig.from_yaml(config_path)
    elif config is None:
        config = ServeConfig()

    app = FastAPI(
        title="LLMRouter Serve",
        description="OpenAI-compatible API with intelligent routing",
        version="1.0.0"
    )

    # Initialize components
    router_adapter = RouterAdapter(
        router_name=config.router_name,
        config_path=config.router_config_path,
        fail_on_error=config.fail_on_routing_error
    )
    llm_backend = LLMBackend(config)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "router": config.router_name,
            "llms": list(config.llms.keys())
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {"id": name, "object": "model"}
                for name in config.llms.keys()
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Extract user query
        user_query = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_query = m["content"][:500]
                break

        # Select model
        available_models = list(config.llms.keys())
        if request.model == "auto" or request.model not in available_models:
            try:
                selected_model = router_adapter.route(user_query, available_models)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"routing failed: {type(e).__name__}: {e}")
            info = router_adapter.last_router_info or {}
            reason = info.get("routing_reason")
            signals = info.get("routing_signals")
            confidence = info.get("routing_confidence")
            judge_ms = info.get("routing_judge_latency_ms")
            extra = ""
            if reason:
                extra += f" reason={reason}"
            if signals:
                extra += f" signals={signals}"
            if confidence is not None:
                extra += f" confidence={confidence}"
            if judge_ms is not None:
                extra += f" judge_ms={judge_ms}"
            print(f"[Router] Query: '{user_query[:50]}...' -> {selected_model}{extra}")
        else:
            selected_model = request.model

        # Call LLM
        if request.stream:
            async def generate():
                first_chunk = True
                async for chunk in llm_backend.call(
                    selected_model, messages, request.max_tokens,
                    request.temperature, stream=True
                ):
                    # Add model prefix
                    if first_chunk and config.show_model_prefix and "content" in chunk:
                        try:
                            data = json.loads(chunk[6:])
                            if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + data["choices"][0]["delta"]["content"]
                                chunk = f"data: {json.dumps(data)}\n\n"
                        except:
                            pass
                        first_chunk = False
                    yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            result = await llm_backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=False
            )

            # Add model prefix
            if config.show_model_prefix and result.get("choices"):
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    result["choices"][0]["message"]["content"] = f"[{selected_model}] {content}"

            result["model"] = selected_model
            result["model"] = selected_model
            return result

    @app.websocket("/v1/chat/ws")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time streaming"""
        await websocket.accept()
        try:
            # Receive request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            # Extract user query
            user_query = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    user_query = m["content"][:500]
                    break

            # Select model
            available_models = list(config.llms.keys())
            if request.model == "auto" or request.model not in available_models:
                selected_model = router_adapter.route(user_query, available_models)
                print(f"[WS Router] Query: '{user_query[:50]}...' -> {selected_model}")
            else:
                selected_model = request.model

            # Call LLM backend in streaming mode
            first_chunk = True
            async for chunk in llm_backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=True
            ):
                # Add model prefix
                if first_chunk and config.show_model_prefix and "content" in chunk:
                    try:
                        data_chunk = json.loads(chunk[6:])
                        if data_chunk.get("choices") and data_chunk["choices"][0].get("delta", {}).get("content"):
                            data_chunk["choices"][0]["delta"]["content"] = f"[{selected_model}] " + data_chunk["choices"][0]["delta"]["content"]
                            chunk = f"data: {json.dumps(data_chunk)}\n\n"
                    except:
                        pass
                    first_chunk = False
                
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        # Send as JSON if it's a valid data chunk
                        json_str = chunk[6:]
                        await websocket.send_json(json.loads(json_str))
                    except:
                        await websocket.send_text(chunk)
                else:
                    await websocket.send_text(chunk)

        except WebSocketDisconnect:
            print("[WS] Client disconnected")
        except Exception as e:
            print(f"[WS Error] {type(e).__name__}: {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

    return app


def run_server(app: FastAPI = None, config_path: str = None, host: str = "0.0.0.0", port: int = 8000):
    """Run server"""
    if app is None:
        app = create_app(config_path=config_path)

    print(f"""
============================================================
  LLMRouter Serve
============================================================
  Server: http://{host}:{port}
  API:    http://{host}:{port}/v1/chat/completions
  Health: http://{host}:{port}/health
============================================================
""")

    uvicorn.run(app, host=host, port=port)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMRouter Serve")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    run_server(config_path=args.config, host=args.host, port=args.port)

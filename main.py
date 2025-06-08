import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from typing import List, Optional,AsyncGenerator
import time

# 定义请求体模型
class OllamaRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False # Ollama API 支持流式响应，这里默认为非流式

# 定义响应体模型 (简化版，只包含响应文本)
class OllamaResponse(BaseModel):
    model: str
    response: str
    done: bool

# 健康检查响应模型
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    ollama_connected: bool
    ollama_url: str

# Ollama 连通性测试响应模型
class OllamaConnectivityResponse(BaseModel):
    connected: bool
    ollama_url: str
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

# 模型信息模型
class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified: Optional[str] = None

# 模型列表响应模型
class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int
    
class StreamChunk(BaseModel):
    model: str
    response: str
    done: bool

app = FastAPI(
    title="Ollama FastAPI Server",
    description="一个使用 FastAPI 与 Ollama 模型进行交互的 API 服务器。",
    version="0.1.0",
)

OLLAMA_API_URL = "http://localhost:11434/api/generate" # Ollama API 的 generate 端点
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API 基础地址

@app.post("/api/generate", response_model=OllamaResponse)
async def generate_text(request: OllamaRequest):
    """
    接收用户请求，调用 Ollama 服务生成文本。
    支持流式和非流式响应。
    """
    if request.stream:
        """
        流式响应处理
        """
        return StreamingResponse(
            )
    else:
        return

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口，检查服务状态和 Ollama 连接状态。
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    ollama_connected = False
    
    # 检查 Ollama 连通性
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_connected = True
    except:
        ollama_connected = False
    
    return HealthResponse(
        status="healthy",
        timestamp=timestamp,
        ollama_connected=ollama_connected,
        ollama_url=OLLAMA_BASE_URL
    )

@app.get("/api/connectivity", response_model=OllamaConnectivityResponse)
async def test_ollama_connectivity():
    """
    测试与 Ollama 服务的连通性。
    """
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                return OllamaConnectivityResponse(
                    connected=True,
                    ollama_url=OLLAMA_BASE_URL,
                    response_time_ms=round(response_time_ms, 2)
                )
            else:
                return OllamaConnectivityResponse(
                    connected=False,
                    ollama_url=OLLAMA_BASE_URL,
                    response_time_ms=round(response_time_ms, 2),
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
    except httpx.RequestError as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return OllamaConnectivityResponse(
            connected=False,
            ollama_url=OLLAMA_BASE_URL,
            response_time_ms=round(response_time_ms, 2),
            error_message=f"连接错误: {str(e)}"
        )
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return OllamaConnectivityResponse(
            connected=False,
            ollama_url=OLLAMA_BASE_URL,
            response_time_ms=round(response_time_ms, 2),
            error_message=f"未知错误: {str(e)}"
        )

@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    """
    获取 Ollama 中可用的模型列表。
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            
            ollama_data = response.json()
            models = []
            
            # 解析 Ollama 返回的模型信息
            if "models" in ollama_data:
                for model_data in ollama_data["models"]:
                    model_info = ModelInfo(
                        name=model_data.get("name", "unknown"),
                        size=model_data.get("size", None),
                        modified=model_data.get("modified_at", None)
                    )
                    models.append(model_info)
            
            return ModelsResponse(
                models=models,
                count=len(models)
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"获取模型列表失败: {e.response.status_code} - {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到 Ollama 服务: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表时发生内部错误: {str(e)}"
        )

@app.get("/api/version")
async def get_ollama_version():
    """
    获取 Ollama 服务的版本信息。
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"获取版本信息失败: {e.response.status_code} - {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到 Ollama 服务: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取版本信息时发生内部错误: {str(e)}"
        )

@app.delete("/api/model/{model_name}")
async def delete_model(model_name: str):
    """
    删除指定的模型。
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {"name": model_name}
            response = await client.delete(f"{OLLAMA_BASE_URL}/api/delete", json=payload)
            response.raise_for_status()
            return {"message": f"模型 '{model_name}' 删除成功"}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"模型 '{model_name}' 未找到"
            )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"删除模型失败: {e.response.status_code} - {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到 Ollama 服务: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除模型时发生内部错误: {str(e)}"
        )

@app.get("/")
async def read_root():
    return {"message": "欢迎使用 Ollama FastAPI 服务器!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
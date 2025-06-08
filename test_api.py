import pytest
import httpx # Required for Request, Response, HTTPStatusError, RequestError
from httpx import AsyncClient, Response as HttpxResponse, Request as HttpxRequest, HTTPStatusError, RequestError
from fastapi import status

# 从你的 main.py 文件导入 app 和可能需要的 Pydantic 模型
from main import app

# 将此模块中的所有测试标记为异步执行
pytestmark = pytest.mark.asyncio

@pytest.fixture
async def client():
    """
    创建一个异步的 HTTP 测试客户端。
    """
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac

async def test_read_root(client: AsyncClient):
    """
    测试根路径 ("/") 是否返回预期的欢迎消息。
    """
    response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "欢迎使用 Ollama FastAPI 服务器!"}

async def test_generate_text_success(client: AsyncClient, mocker):
    """
    测试 /api/generate 端点在 Ollama 成功响应时的行为。
    """
    # 模拟 Ollama API 的成功响应
    mock_ollama_api_response_data = {
        "model": "test-model-from-ollama",
        "response": "这是来自 Ollama 的测试响应。",
        "done": True,
        "created_at": "2023-01-01T00:00:00Z", # Ollama 响应中可能包含的其他字段
        "context": [1, 2, 3]
    }
    
    mock_http_response = mocker.Mock(spec=HttpxResponse)
    mock_http_response.status_code = status.HTTP_200_OK
    mock_http_response.json.return_value = mock_ollama_api_response_data
    mock_http_response.raise_for_status = mocker.Mock() # 对于成功的响应，raise_for_status 不应抛出异常

    # 修补 main.py 中 httpx.AsyncClient().post 方法
    mocker.patch("main.httpx.AsyncClient.post", return_value=mock_http_response)

    request_payload = {"model": "requested-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_200_OK
    response_data = api_response.json()
    assert response_data["model"] == mock_ollama_api_response_data["model"]
    assert response_data["response"] == mock_ollama_api_response_data["response"]
    assert response_data["done"] == mock_ollama_api_response_data["done"]

async def test_generate_text_ollama_model_not_found(client: AsyncClient, mocker):
    """
    测试当 Ollama API 返回 404 (模型未找到) 时的错误处理。
    """
    # 模拟 Ollama API 返回 404 错误
    # 创建一个模拟的 httpx.Request 对象
    mock_request_obj = mocker.Mock(spec=HttpxRequest)
    mock_request_obj.method = "POST"
    mock_request_obj.url = "http://localhost:11434/api/generate"

    mock_ollama_error_response = mocker.Mock(spec=HttpxResponse)
    mock_ollama_error_response.status_code = status.HTTP_404_NOT_FOUND
    mock_ollama_error_response.text = "Ollama error: model 'non-existent-model' not found"
    
    # raise_for_status 应该抛出 HTTPStatusError
    http_status_error = HTTPStatusError(
        message=f"Client error '404 Not Found' for url '{mock_request_obj.url}'",
        request=mock_request_obj,
        response=mock_ollama_error_response
    )
    mock_ollama_error_response.raise_for_status = mocker.Mock(side_effect=http_status_error)

    mocker.patch("main.httpx.AsyncClient.post", return_value=mock_ollama_error_response)

    request_payload = {"model": "non-existent-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Ollama 模型 'non-existent-model' 未找到" in api_response.json()["detail"]

async def test_generate_text_ollama_request_error(client: AsyncClient, mocker):
    """
    测试当连接到 Ollama API 失败时的错误处理 (例如，服务未运行)。
    """
    # 模拟 httpx.RequestError (例如，连接错误)
    # 创建一个模拟的 httpx.Request 对象
    mock_request_obj = mocker.Mock(spec=HttpxRequest)
    mock_request_obj.method = "POST"
    mock_request_obj.url = "http://localhost:11434/api/generate"

    request_error = RequestError(message="Connection refused", request=mock_request_obj)
    mocker.patch("main.httpx.AsyncClient.post", side_effect=request_error)

    request_payload = {"model": "any-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "无法连接到 Ollama 服务" in api_response.json()["detail"]

async def test_generate_text_ollama_malformed_response(client: AsyncClient, mocker):
    """
    测试当 Ollama API 返回成功状态码但响应体格式不正确时的错误处理。
    """
    mock_ollama_malformed_data = {
        "model": "test-model",
        # "response": "此字段缺失", # 故意缺少 'response' 字段
        "done": True
    }
    mock_http_response = mocker.Mock(spec=HttpxResponse)
    mock_http_response.status_code = status.HTTP_200_OK
    mock_http_response.json.return_value = mock_ollama_malformed_data
    mock_http_response.raise_for_status = mocker.Mock()

    mocker.patch("main.httpx.AsyncClient.post", return_value=mock_http_response)

    request_payload = {"model": "test-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "从 Ollama 收到的响应格式不正确" in api_response.json()["detail"]

async def test_generate_text_ollama_other_http_error(client: AsyncClient, mocker):
    """
    测试当 Ollama API 返回其他 HTTP 错误 (例如 400 Bad Request) 时的处理。
    """
    mock_request_obj = mocker.Mock(spec=HttpxRequest)
    mock_request_obj.method = "POST"
    mock_request_obj.url = "http://localhost:11434/api/generate"

    mock_ollama_error_response = mocker.Mock(spec=HttpxResponse)
    mock_ollama_error_response.status_code = status.HTTP_400_BAD_REQUEST
    mock_ollama_error_response.text = "Ollama error: bad request"
    
    http_status_error = HTTPStatusError(
        message=f"Client error '400 Bad Request' for url '{mock_request_obj.url}'",
        request=mock_request_obj,
        response=mock_ollama_error_response
    )
    mock_ollama_error_response.raise_for_status = mocker.Mock(side_effect=http_status_error)

    mocker.patch("main.httpx.AsyncClient.post", return_value=mock_ollama_error_response)

    request_payload = {"model": "some-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_400_BAD_REQUEST
    assert "向 Ollama 服务发送的请求无效" in api_response.json()["detail"]
    assert "Ollama error: bad request" in api_response.json()["detail"]

async def test_generate_text_unexpected_exception(client: AsyncClient, mocker):
    """
    测试在请求处理过程中发生意外的非 HTTPX 异常时的错误处理。
    """
    mocker.patch("main.httpx.AsyncClient.post", side_effect=ValueError("发生了意外的内部值错误"))

    request_payload = {"model": "any-model", "prompt": "你好"}
    api_response = await client.post("/api/generate", json=request_payload)

    assert api_response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "处理请求时发生内部错误: 发生了意外的内部值错误" in api_response.json()["detail"]
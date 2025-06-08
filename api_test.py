import requests
import json

# FastAPI 服务器地址
BASE_URL = "http://localhost:8000"

def test_get_request():
    """
    测试 GET 请求 - 访问根路径
    """
    print("=== 测试 GET 请求 ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        if response.status_code == 200:
            print("✅ GET 请求成功\n")
        else:
            print("❌ GET 请求失败\n")
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保 FastAPI 服务器正在运行\n")
    except Exception as e:
        print(f"❌ 请求出错: {e}\n")

def test_post_request():
    """
    测试 POST 请求 - 文本生成
    """
    print("=== 测试 POST 请求 ===")
    
    # 请求数据
    data = {
        "model": "deepseek-r1:1.5b",  # 你可以根据实际可用模型修改
        "prompt": "你好，请说一句话",
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=data,
            timeout=60  # 设置超时时间
        )
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("✅ POST 请求成功\n")
        else:
            print("❌ POST 请求失败\n")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保 FastAPI 服务器正在运行\n")
    except requests.exceptions.Timeout:
        print("❌ 请求超时\n")
    except Exception as e:
        print(f"❌ 请求出错: {e}\n")

def main():
    print("开始测试 FastAPI 接口...\n")
    
    # 测试 GET 请求
    test_get_request()
    
    # 测试 POST 请求
    test_post_request()
    
    print("测试完成！")

if __name__ == "__main__":
    main()
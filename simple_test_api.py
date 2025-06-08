import asyncio
import httpx
import json

# FastAPI 应用的地址
BASE_URL = "http://localhost:8000"

async def test_api_endpoints():
    """
    测试所有的 API 端点
    """
    print("=== Ollama FastAPI 服务器测试 ===\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # 1. 测试根路径
        print("1. 测试根路径 (GET /)")
        print("-" * 40)
        try:
            response = await client.get(f"{BASE_URL}/")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.json()}")
            print("✅ 根路径测试通过\n" if response.status_code == 200 else "❌ 根路径测试失败\n")
        except Exception as e:
            print(f"❌ 请求失败: {e}\n")

        # 2. 测试健康检查
        print("2. 测试健康检查 (GET /health)")
        print("-" * 40)
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"状态码: {response.status_code}")
            response_data = response.json()
            print(f"响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            ollama_status = "连接成功" if response_data.get("ollama_connected") else "连接失败"
            print(f"Ollama 状态: {ollama_status}")
            print("✅ 健康检查测试通过\n" if response.status_code == 200 else "❌ 健康检查测试失败\n")
        except Exception as e:
            print(f"❌ 请求失败: {e}\n")

        # 3. 测试连通性
        print("3. 测试 Ollama 连通性 (GET /api/connectivity)")
        print("-" * 40)
        try:
            response = await client.get(f"{BASE_URL}/api/connectivity")
            print(f"状态码: {response.status_code}")
            response_data = response.json()
            print(f"响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            if response_data.get("connected"):
                print(f"✅ Ollama 连接成功，响应时间: {response_data.get('response_time_ms')}ms")
            else:
                print(f"❌ Ollama 连接失败: {response_data.get('error_message')}")
            print()
        except Exception as e:
            print(f"❌ 请求失败: {e}\n")

        # 4. 测试获取模型列表
        print("4. 测试获取模型列表 (GET /api/models)")
        print("-" * 40)
        try:
            response = await client.get(f"{BASE_URL}/api/models")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                print(f"模型数量: {response_data.get('count', 0)}")
                models = response_data.get('models', [])
                if models:
                    print("可用模型:")
                    for model in models[:3]:  # 只显示前3个模型
                        print(f"  - {model.get('name')} (大小: {model.get('size', 'N/A')})")
                    if len(models) > 3:
                        print(f"  ... 还有 {len(models) - 3} 个模型")
                else:
                    print("  没有找到可用模型")
                print("✅ 获取模型列表成功")
            else:
                error_detail = response.json().get('detail', '未知错误')
                print(f"❌ 获取模型列表失败: {error_detail}")
            print()
        except Exception as e:
            print(f"❌ 请求失败: {e}\n")

        # 5. 测试获取版本信息
        print("5. 测试获取版本信息 (GET /api/version)")
        print("-" * 40)
        try:
            response = await client.get(f"{BASE_URL}/api/version")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                version_data = response.json()
                print(f"版本信息: {json.dumps(version_data, indent=2, ensure_ascii=False)}")
                print("✅ 获取版本信息成功")
            else:
                error_detail = response.json().get('detail', '未知错误')
                print(f"❌ 获取版本信息失败: {error_detail}")
            print()
        except Exception as e:
            print(f"❌ 请求失败: {e}\n")

        # 6. 测试文本生成（需要先获取可用模型）
        print("6. 测试文本生成 (POST /api/generate)")
        print("-" * 40)
        
        # 首先尝试获取可用模型
        available_model = None
        try:
            models_response = await client.get(f"{BASE_URL}/api/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = models_data.get('models', [])
                if models:
                    available_model = models[0]['name']  # 使用第一个可用模型
                    print(f"使用模型: {available_model}")
        except:
            pass
        
        if not available_model:
            # 如果没有获取到模型，使用常见的模型名称进行测试
            test_models = ["llama3", "llama2", "mistral", "gemma"]
            print("未找到可用模型，将尝试常见模型名称...")
            
            for test_model in test_models:
                print(f"\n尝试模型: {test_model}")
                payload = {
                    "model": test_model,
                    "prompt": "你好，请简单介绍一下你自己，不超过20个字。",
                    "stream": False
                }
                
                try:
                    response = await client.post(f"{BASE_URL}/api/generate", json=payload)
                    print(f"状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"模型: {response_data.get('model')}")
                        print(f"响应: {response_data.get('response')}")
                        print(f"完成: {response_data.get('done')}")
                        print(f"✅ 文本生成测试成功 (模型: {test_model})")
                        break
                    else:
                        error_detail = response.json().get('detail', '未知错误')
                        print(f"❌ 测试失败: {error_detail}")
                        if "未找到" in error_detail:
                            continue  # 尝试下一个模型
                        else:
                            break  # 其他错误，不再尝试
                            
                except Exception as e:
                    print(f"❌ 请求失败: {e}")
                    continue
            else:
                print("❌ 所有测试模型都不可用")
        else:
            # 使用获取到的可用模型进行测试
            payload = {
                "model": available_model,
                "prompt": "你好，请简单介绍一下你自己，不超过20个字。",
                "stream": False
            }
            
            try:
                response = await client.post(f"{BASE_URL}/api/generate", json=payload)
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"模型: {response_data.get('model')}")
                    print(f"响应: {response_data.get('response')}")
                    print(f"完成: {response_data.get('done')}")
                    print("✅ 文本生成测试成功")
                else:
                    error_detail = response.json().get('detail', '未知错误')
                    print(f"❌ 文本生成测试失败: {error_detail}")
                    
            except Exception as e:
                print(f"❌ 请求失败: {e}")
        
        print()

        # 7. 测试错误处理 - 使用不存在的模型
        print("7. 测试错误处理 - 不存在的模型")
        print("-" * 40)
        payload = {
            "model": "non_existent_model_12345",
            "prompt": "测试错误处理",
            "stream": False
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/generate", json=payload)
            print(f"状态码: {response.status_code}")
            if response.status_code == 404:
                error_detail = response.json().get('detail', '未知错误')
                print(f"错误信息: {error_detail}")
                print("✅ 错误处理测试通过 (正确返回404)")
            else:
                print("❌ 错误处理测试失败 (未返回预期的404)")
        except Exception as e:
            print(f"❌ 请求失败: {e}")
        
        print()

        # 8. 测试删除模型 (可选，谨慎使用)
        print("8. 测试删除模型 (跳过 - 避免意外删除)")
        print("-" * 40)
        print("⚠️  删除模型测试已跳过，避免意外删除重要模型")
        print("如需测试，请手动取消注释相关代码")
        
        # 如果你想测试删除功能，请取消下面的注释并确保使用测试模型
        # test_model_to_delete = "test_model_that_can_be_deleted"
        # try:
        #     response = await client.delete(f"{BASE_URL}/api/model/{test_model_to_delete}")
        #     print(f"状态码: {response.status_code}")
        #     if response.status_code == 200:
        #         result = response.json()
        #         print(f"结果: {result}")
        #         print("✅ 删除模型测试成功")
        #     else:
        #         error_detail = response.json().get('detail', '未知错误')
        #         print(f"❌ 删除模型测试失败: {error_detail}")
        # except Exception as e:
        #     print(f"❌ 请求失败: {e}")
        
        print()

    print("=== 测试完成 ===")
    print("\n使用说明:")
    print("1. 确保 FastAPI 服务器正在运行 (python main.py)")
    print("2. 确保 Ollama 服务正在运行")
    print("3. 如果文本生成测试失败，请检查 Ollama 中是否有可用模型")
    print("4. 可以通过 'ollama list' 命令查看本地可用模型")

if __name__ == "__main__":
    print("开始测试 Ollama FastAPI 服务器...\n")
    asyncio.run(test_api_endpoints())
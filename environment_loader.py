# environment_loader.py
import os

from dotenv import load_dotenv


# 加载环境变量
def load_environment() -> None:
    load_dotenv()


# 检查并返回API密钥，如果未设置则抛出错误
def check_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY in the .env file")
    return api_key

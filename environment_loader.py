# environment_loader.py
import os

from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv()


def check_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY in the .env file")
    return api_key

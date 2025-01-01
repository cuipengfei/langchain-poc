# chat_model_setup.py
from typing import Dict

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel


def create_chat_model() -> BaseChatModel:
    return ChatTongyi(model="qwen-max", streaming=False)


def get_session_history(store: Dict[str, BaseChatMessageHistory], session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

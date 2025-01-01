# chat_model_setup.py

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel


# 创建聊天模型实例
def create_chat_model() -> BaseChatModel:
    return ChatTongyi(model="qwen-max", streaming=False)


store = {}


# 获取会话历史记录，如果会话ID不存在则创建新的历史记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

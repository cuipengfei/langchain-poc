# token_counter.py
from typing import List

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage


# 计算字符串的token数量
def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


# 计算消息列表的token数量
def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens: int = 3
    tokens_per_message: int = 3
    tokens_per_name: int = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role: str = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
                tokens_per_message
                + str_token_counter(role)
                + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens

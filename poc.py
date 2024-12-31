import http
import http.client
import logging
import os
from operator import itemgetter
from typing import List

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


def setup_logging():
    logging.basicConfig(
        format="%(levelname)s [%(asctime)s] %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    httpclient_logger = logging.getLogger("http.client")

    def httpclient_log(*args):
        httpclient_logger.log(logging.DEBUG, " ".join(args))

    http.client.print = httpclient_log
    http.client.HTTPConnection.debuglevel = 1

    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.DEBUG)

    for handler in urllib3_logger.handlers:
        handler.setLevel(logging.DEBUG)


def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
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


if __name__ == "__main__":
    setup_logging()

    # 从.env文件加载API key
    load_dotenv()

    # 使用os.getenv获取API key，并提供错误处理
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("请在.env文件中设置 DASHSCOPE_API_KEY")

    # 创建聊天模型
    chatLLM = ChatTongyi(
        model="qwen-max",
        streaming=False,
    )

    # 添加聊天历史存储
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]


    # 创建向量存储和检索器
    vectorstore = Chroma(
        collection_name="ai_learning",
        embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
        persist_directory="vectordb"
    )
    retriever = vectorstore.as_retriever(search_type="similarity")

    # 创建消息修剪器
    trimmer = trim_messages(
        max_tokens=4096,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
    )

    # 创建聊天模型链
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks. 
                Always respond in Chinese, no matter the language of the input.
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise.
                Context: {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    context = itemgetter("question") | retriever | format_docs
    first_step = RunnablePassthrough.assign(context=context)
    chain = first_step | prompt | trimmer | chatLLM

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 配置会话ID
    config = {"configurable": {"session_id": "default_session"}}

    # 运行聊天模型链
    while True:
        user_input = input("You:> ")
        if user_input.lower() == 'exit':
            break
        if user_input.strip() == "":
            continue
        try:
            stream = with_message_history.stream({"question": user_input}, config=config)
            for chunk in stream:
                print(chunk.content, end='', flush=True)
            print()
        except ValueError as e:
            print(f"Error: {e}")

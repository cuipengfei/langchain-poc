# poc.py
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_model_setup import create_chat_model, get_session_history
from environment_loader import load_environment, check_api_key
from logging_setup import setup_logging
from token_counter import tiktoken_counter
from vector_store_setup import create_vector_store


# 格式化文档内容为字符串
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    setup_logging()  # 设置日志记录
    load_environment()  # 加载环境变量
    api_key = check_api_key()  # 检查API密钥

    chatLLM = create_chat_model()  # 创建聊天模型实例

    vectorstore = create_vector_store()  # 创建向量存储实例
    retriever = vectorstore.as_retriever(search_type="similarity")  # 创建检索器

    # 修剪消息以确保总token数不超过4096
    trimmer = trim_messages(
        max_tokens=4096,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
    )

    # 创建聊天提示模板
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

    # 定义上下文处理流程
    context = RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    first_step = RunnablePassthrough.assign(context=context)
    chain = first_step | prompt | trimmer | chatLLM

    # 创建带有消息历史记录的可运行对象
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 主循环，处理用户输入
    while True:
        user_input: str = input("You:> ")
        if user_input.lower() == 'exit':
            break
        if user_input.strip() == "":
            continue
        try:
            stream = with_message_history.stream(
                input={"question": user_input},
                config={"configurable": {"session_id": "default_session"}}
            )
            for chunk in stream:
                print(chunk.content, end='', flush=True)
            print()
        except ValueError as e:
            print(f"Error: {e}")

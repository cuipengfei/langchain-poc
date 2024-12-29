import http
import http.client
import logging
import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory


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


def get_multiline_input(prompt='请输入您的问题（或输入 "exit" 退出）：', end_marker='end'):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'exit':
                return 'exit'
            if line.strip().lower() == end_marker.lower():
                break
            lines.append(line)
        except EOFError:  # Ctrl+D (Unix) 或 Ctrl+Z (Windows)
            break
    return '\n'.join(lines).strip()  # 确保返回的内容不包含多余的空白字符


if __name__ == "__main__":
    setup_logging()

    # 从.env文件加载API key
    load_dotenv()

    # 使用os.getenv获取API key，并提供错误处理
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
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


    # 将聊天模型与历史记录包装在一起
    with_message_history = RunnableWithMessageHistory(chatLLM, get_session_history)

    # 配置会话ID
    config = {"configurable": {"session_id": "default_session"}}

    while True:
        try:
            user_input = get_multiline_input()
            if user_input.lower() == 'exit':
                print("程序已退出。")
                break

            # 使用带历史记录的方式调用模型
            res = with_message_history.invoke(
                [HumanMessage(content=user_input)],
                config=config
            )
            print("Qwen的回答：", res.content)

        except Exception as e:
            print(f"发生错误: {e}")

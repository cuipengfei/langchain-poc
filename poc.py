# poc.py
from operator import itemgetter
from typing import Dict, List

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


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    setup_logging()
    load_environment()
    api_key = check_api_key()

    chatLLM = create_chat_model()

    vectorstore = create_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity")

    trimmer = trim_messages(
        max_tokens=4096,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
    )

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

    context = RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    first_step = RunnablePassthrough.assign(context=context)
    chain = first_step | prompt | trimmer | chatLLM

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda session_id: get_session_history({}, session_id),
        input_messages_key="question",
        history_messages_key="history",
    )

    config: Dict[str, Dict[str, str]] = {"configurable": {"session_id": "default_session"}}

    while True:
        user_input: str = input("You:> ")
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

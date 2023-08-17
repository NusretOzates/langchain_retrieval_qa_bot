import json
import os
from typing import List

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from redis import Redis

from data_loaders import create_index, load_pdf_file, load_text_file, load_website
from memory_loader import load_memory
from prompts import QUESTION_CREATOR_TEMPLATE, ASSISTANT_PROMPT
from redis_ops import clear_user, save_files, get_files, add_qa_pair, check_user_exist


@st.cache_resource
def load_redis_connection() -> Redis:
    """Loads the Redis connection

    Returns:
        A Redis object.
    """
    return Redis(host="localhost", port=6379, decode_responses=True)


@st.cache_resource
def load_chain(
    file_names: List[str], file_types: [str]
) -> ConversationalRetrievalChain:
    """Loads the ConversationalRetrievalChain

    Args:
        file_name: Name of the file
        file_type: Type of the file

    Returns:
        A ConversationalRetrievalChain object.
    """
    all_docs = []

    for file_name, file_type in zip(file_names, file_types):

        match file_type:
            case "text/plain":
                doc = load_text_file(file_name)
                all_docs.append(doc)
            case "application/pdf":
                doc = load_pdf_file(file_name)
                all_docs.extend(doc)
            case "text/html":
                doc = load_website(file_name)
                all_docs.extend(doc)
            case _:
                st.write("File type is not supported!")
                st.stop()

    retriever = create_index(all_docs)
    condense_question_prompt = PromptTemplate.from_template(QUESTION_CREATOR_TEMPLATE)
    assistant_prompt = PromptTemplate(
        template=ASSISTANT_PROMPT, input_variables=["context", "question"]
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        condense_question_prompt=condense_question_prompt,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt":assistant_prompt},
    )

    return chain


os.environ["OPENAI_API_KEY"] = "sk"
st.set_page_config(layout="wide")
st.title("💬 QA Chatbot")

redis_connection = load_redis_connection()

username = st.text_input("Username")


def get_response(
    chain: ConversationalRetrievalChain,
    question: str,
    memory: ConversationBufferWindowMemory,
) -> str:
    response = chain(
        {
            "question": question,
            "chat_history": memory.load_memory_variables({})["history"],
        }
    )
    answer = response["answer"]
    documents = response["source_documents"]
    answer += "\n\n" + "\n\n".join([f"Source: {doc.metadata}" for doc in documents])

    return answer


if username:
    st.button("Delete all files", on_click=lambda: clear_user(redis_connection,username))
    # Check if we have a key for this user
    if check_user_exist(redis_connection, username):
        # Get a file
        uploaded_files = st.file_uploader(
            "Choose a file", type=["txt", "pdf"], accept_multiple_files=True
        )
        st.header("Or you can give a url")
        url = st.text_input("Url to parse")

        if uploaded_files or url != "":
            # Clear previous files and urls
            redis_connection.delete(f"{username}_filenames")
            redis_connection.delete(f"{username}_filetypes")

            if not os.path.exists(username):
                os.mkdir(username)

            if uploaded_files:
                for file in uploaded_files:
                    # Save the file
                    save_path = os.path.join(username, file.name)
                    print(f"Saving file to: {save_path}")

                    with open(save_path, "wb") as f:
                        f.write(file.read())

                names, types = zip(
                    *[
                        (os.path.join(username, file.name), file.type)
                        for file in uploaded_files
                    ]
                )

                save_files(redis_connection, username, names, types)
                chain = load_chain(names, types)
            else:
                # Save url to redis
                save_files(username, [url], ["text/html"])
                chain = load_chain([url], ["text/html"])

            # Create a new key
            redis_connection.set(username, 1)

    else:
        # Get the file names and types
        file_names, file_types = get_files(redis_connection, username)

        if not file_names:
            st.error("Something went wrong! Please refresh the page.")
            clear_user(redis_connection, username)
            os.removedirs(username)
            st.stop()

        # Load the chain
        chain = load_chain(file_names, file_types)

        st.info(
            f"Welcome back! You can continue from where you left off. We loaded your previous files/url: {file_names}"
        )

    # Load the memory
    memory, last_user_message = load_memory(redis_connection, username)

    for message in memory.load_memory_variables({})["history"]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
            continue
        st.chat_message("assistant").write(message.content)

    if question := st.chat_input():
        # Get and save the question
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            # Get an answer using question and the conversation history
            answer = get_response(chain, question, memory)

        st.chat_message("assistant").write(answer)

        add_qa_pair(redis_connection, username, question, answer)

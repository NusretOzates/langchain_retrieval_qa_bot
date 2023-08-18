# Set logging for the queries
import logging
import os
from typing import List, Tuple

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from data_loaders import create_index, load_pdf_file, load_text_file
from prompts import ASSISTANT_PROMPT, QUESTION_CREATOR_TEMPLATE
from utils import clear_user, load_memory

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


@st.cache_resource
def load_chain(files: List[Tuple[str, str]]) -> BaseConversationalRetrievalChain:
    """Loads the ConversationalRetrievalChain

    Args:
        files: A list of tuples that contains the file name and the file type.

    Returns:
        A ConversationalRetrievalChain object.
    """
    all_docs = []

    for file_name, file_type in files:
        match file_type:
            case "text/plain":
                doc = load_text_file(file_name)
            case "application/pdf":
                doc = load_pdf_file(file_name)
            case _:
                st.write("File type is not supported!")
                st.stop()

        all_docs.extend(doc)

    retriever = create_index(all_docs)

    condense_question_prompt = PromptTemplate.from_template(QUESTION_CREATOR_TEMPLATE)
    assistant_prompt = PromptTemplate.from_template(ASSISTANT_PROMPT)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        condense_question_prompt=condense_question_prompt,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": assistant_prompt},
    )

    return chain


# In .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
st.set_page_config(layout="wide")
st.title("💬 QA Chatbot")


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


st.button("Delete all files", on_click=clear_user("user_files"))

# Get a file
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file:
    if not os.path.exists("user_files"):
        os.mkdir("user_files")

    # Save the file
    save_path = os.path.join("user_files", uploaded_file.name)
    print(f"Saving file to: {save_path}")

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    chain = load_chain(
        [(os.path.join("user_files", uploaded_file.name), uploaded_file.type)]
    )

    # Load the memory
    memory = load_memory(st)

    if question := st.chat_input():
        # Get and save the question
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            # Get an answer using question and the conversation history
            answer = get_response(chain, question, memory)

        st.chat_message("assistant").write(answer)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

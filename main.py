import streamlit as st
from data_loaders import load_text_files, load_pdf_files, load_website, create_index
from prompts import QUESTION_CREATOR_TEMPLATE
from memory_loader import load_memory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os


@st.cache_resource
def load_chain(file_name: str, file_type: str):
    if file_type == "text/plain":
        docs = load_text_files([file_name])
    elif file_type == "application/pdf":
        docs = load_pdf_files([file_name])
    elif file_type == "text/html":
        docs = load_website(file_name)
    else:
        st.write("File type is not supported!")
        st.stop()

    retriever = create_index(docs)
    condense_question_prompt = PromptTemplate.from_template(QUESTION_CREATOR_TEMPLATE)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        condense_question_prompt=condense_question_prompt,
        verbose=True,
    )

    return chain


os.environ["OPENAI_API_KEY"] = "sk-..."
st.set_page_config(layout="wide")
st.title("💬 QA Chatbot")

# Get a file
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
st.header("Or you can give a url")
url = st.text_input("Url to parse")


if uploaded_file is not None or url:
    if uploaded_file:
        # Save the file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        chain = load_chain(uploaded_file.name, uploaded_file.type)
    else:
        chain = load_chain(url, "text/html")

    # Load the memory
    memory = load_memory(st)

    st.write("## 🤖 Chatbot is ready to answer your questions!")

    if question := st.chat_input():
        # Get and save the question
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Get an answer using question and the conversation history
        response = chain(
            {
                "question": question,
                "chat_history": memory.load_memory_variables({})["history"],
            }
        )
        answer = response["answer"]
        # Save the answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

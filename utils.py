import os
import shutil
from langchain.memory import ConversationBufferWindowMemory


def clear_user(username: str):
    """Clear user files
    Args:
        username: Username

    Returns:
        None

    """
    # If the user has a folder, delete it
    if os.path.exists(username):
        shutil.rmtree(username)


def load_memory(st) -> ConversationBufferWindowMemory:
    """Load memory from session state

    Args:
        st: streamlit object

    Returns:
        memory_loader: ConversationBufferMemory object
    """
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi!"}]

    for index, msg in enumerate(st.session_state.messages):
        st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "user" and index < len(st.session_state.messages) - 1:
            user_input = msg["content"]
            assistant_output = st.session_state.messages[index + 1]["content"]

            memory.save_context(
                {"input": user_input},
                {"output": assistant_output},
            )

    return memory

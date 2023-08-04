import json
from typing import Optional, Tuple

from langchain.memory import ConversationBufferWindowMemory
from redis import Redis


def load_memory(
    redis_coonection: Redis, username: str
) -> Tuple[ConversationBufferWindowMemory, Optional[str]]:
    """Load memory from redis

    Args:
        redis_coonection: redis connection

    Returns:
        memory_loader: ConversationBufferMemory object
        user_message: user message
    """
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    messages = redis_coonection.lrange(f"{username}_messages", 0, -1)
    user_message = ""
    for index, msg in enumerate(messages):
        msg = json.loads(msg)

        if msg["role"] == "user":
            user_message = msg["content"]
            continue

        memory.save_context(
            {"input": user_message},
            {"output": msg["content"]},
        )
        user_message = ""

    return memory, user_message

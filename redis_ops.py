from redis import Redis
from typing import List, Tuple
import json


def clear_user(redis_connection: Redis, username: str):
    """Clear user from redis

    Clear user from redis. This is used to clear the user's files and memory.

    Args:
        redis_connection: Redis connection
        username: Username

    Returns:
        None

    """
    redis_connection.delete(username)
    redis_connection.delete(f"{username}_filenames")
    redis_connection.delete(f"{username}_filetypes")
    redis_connection.delete(f"{username}_messages")


def save_files(
    redis_connection: Redis,
    username: str,
    uploaded_files: List[str],
    uploaded_types: List[str],
):
    """Save filenames and filetypes to redis

    Save filenames and filetypes to redis to restore them later. This is used to keep track of the files that the user
    has uploaded.

    Args:
        redis_connection: Redis connection
        username: Username
        uploaded_files: List of uploaded files
        uploaded_types: List of uploaded filetypes

    Returns:
        None
    """
    redis_connection.rpush(f"{username}_filenames", *uploaded_files)
    redis_connection.rpush(f"{username}_filetypes", *uploaded_types)

    redis_connection.expire(f"{username}_filenames", 1800)
    redis_connection.expire(f"{username}_filetypes", 1800)


def get_files(redis_connection: Redis, username: str) -> Tuple[List[str], List[str]]:
    """Get filenames and filetypes from redis

    Get filenames and filetypes from redis. This is used to restore the files that the user has uploaded.

    Args:
        redis_connection: Redis connection
        username: Username

    Returns:
        List of filenames and filetypes
    """
    filenames = redis_connection.lrange(f"{username}_filenames", 0, -1)
    filetypes = redis_connection.lrange(f"{username}_filetypes", 0, -1)

    if not filenames:
        return [], []

    return filenames, filetypes


def add_qa_pair(redis_connection: Redis, username: str, question: str, answer: str):
    """Add a question and answer pair to redis

    Add a question and answer pair to redis. This is used to keep track of the user's questions and answers.

    Args:
        redis_connection: Redis connection
        username: Username
        question: Question
        answer: Answer

    Returns:
        None
    """

    redis_connection.rpush(
        f"{username}_messages", json.dumps({"role": "user", "content": question})
    )

    # Save the answer
    redis_connection.rpush(
        f"{username}_messages", json.dumps({"role": "assistant", "content": answer})
    )

    # Expire the messages after 30 mins
    redis_connection.expire(f"{username}_messages", 1800)


def check_user_exist(redis_connection: Redis, username: str) -> bool:
    """Check if the user exists in redis

    Args:
        redis_connection:  Redis connection
        username: Username

    Returns:
        True if the user exists, False otherwise
    """

    return not redis_connection.exists(username) and not redis_connection.exists(
        f"{username}_filenames"
    )

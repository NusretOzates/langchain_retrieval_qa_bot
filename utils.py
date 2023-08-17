import os
import shutil

def clear_user(username: str):
    """Clear user files

    Clear user files

    Args:
        username: Username

    Returns:
        None

    """
    # If the user has a folder, delete it
    if os.path.exists(username):
        shutil.rmtree(username)
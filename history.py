import json
import os
import time
from pathlib import Path
from collections import deque


# Load chat history
def load_chat(session_token):
    # Create directory if it doesn't exist
    os.makedirs("./history", exist_ok=True)
    
    file_path = f"./history/chat_history_{session_token}.json"
    try:
        with open(file_path, "r") as f:
            chat_history = deque(json.load(f), maxlen=10)
            return chat_history
    except FileNotFoundError:
        chat_history = deque(maxlen=10)
        return chat_history  # Return empty deque instead of None

def add_message(chat_history, sender, message):
    chat_history.append({"role": sender, "content": message})
    return chat_history

# Save chat history
def save_chat(chat_history, session_token):
    # Create directory if it doesn't exist
    os.makedirs("./history", exist_ok=True)    
    file_path = f"./history/chat_history_{session_token}.json"
    with open(file_path, "w") as f:
        json.dump(list(chat_history), f, indent=4)  # Convert deque to list before saving

def history_cleanup():
    try:
        directory = Path('./history')
        if not directory.exists():
            print("History directory does not exist. Creating it...")
            directory.mkdir(parents=True, exist_ok=True)
            return

        one_day_ago = time.time() - (24 * 60 * 60)  # 24 hours in seconds
        print(f"\nRunning history cleanup at {time.ctime()}")
        print(f"Current time: {time.time()}")
        print(f"Files older than: {time.ctime(one_day_ago)} will be deleted")
        
        deleted_count = 0
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() == '.json':
                last_modified = file.stat().st_mtime
                if last_modified <= one_day_ago:
                    print(f"Deleting {file.name} (last modified: {time.ctime(last_modified)})")
                    file.unlink()
                    deleted_count += 1
        
        print(f"Cleanup complete. Deleted {deleted_count} files.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


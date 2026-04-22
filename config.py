import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "LLM_Redial" / "Movie"

ITEM_MAP_PATH = DATA_DIR / "item_map.json"
USER_IDS_PATH = DATA_DIR / "user_ids.json"
FINAL_DATA_PATH = DATA_DIR / "final_data.jsonl"
CONVERSATION_PATH = DATA_DIR / "Conversation.txt"

QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

EMBEDDING_BASE_URL = "http://192.168.1.12:1234/v1"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

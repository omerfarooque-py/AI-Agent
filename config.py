# config.py
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "./faiss_db"
PDF_CHUNK_SIZE = 1000
PDF_CHUNK_OVERLAP = 200
MODEL_MAP = {
    "Speed": "llama-3.1-8b-instant",
    "Expert": "llama-3.3-70b-versatile"
}

import os
print(os.listdir())
print(os.path.exists("./faiss_db"))
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community import vectorstores
from config import PERSIST_DIR, EMBED_MODEL
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def is_cache():
   return os.path.exists(PERSIST_DIR) and len(os.listdir()) > 0

def get_vectorstore():
   if is_cache:
    return FAISS.load_local(
        PERSIST_DIR,
        embeddings,
        allow_dangerous_deserialization=True    
    )
   
   if splits:
     vectorstore = FAISS.from_documents(splits, documents)
     vectorstore.save_local(PERSIST_DIR)
     return vectorstore
   return None



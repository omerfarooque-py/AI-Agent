import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, PERSIST_DIR

embeddings =  OllamaEmbeddings(model=EMBED_MODEL)


def load_vectorstore():
    if os.path.exists(PERSIST_DIR):
        return FAISS.load_local(
            PERSIST_DIR,
            embeddings,
            allow_dangerous_deserialization= True
        )
    return None

def save_vectorstore(vectorstore):
    vectorstore.save_local(PERSIST_DIR)

def create_vectorstore_from_docs(docs):
    return FAISS.from_documents(docs, embeddings)    
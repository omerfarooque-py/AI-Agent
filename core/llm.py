from langchain_groq import ChatGroq
from config import MODEL_MAP



def build_llm(api_key: str, model_type: str = "Speed"):
    model_id = MODEL_MAP.get(model_type, MODEL_MAP["Speed"])
    return ChatGroq(
        groq_api_key = api_key,
        model_name = model_id,
        temperature = 0,
        streaming=True
    )

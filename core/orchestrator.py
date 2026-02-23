from core.utils import convert_chat_history
from .rag_engine import RAG_TEMPLATE


def run_controlled_rag(question, rag_chain, web_func, chat_history):

    # 1. Get PDF Context
    # We call the retriever directly to get documents without the LLM answering yet
    pdf_docs = rag_chain.retriever.invoke(question)
    pdf_context = "\n".join([d.page_content for d in pdf_docs])
    pdf_sources = [d.metadata.get("source") for d in pdf_docs]

    # 2. Get Web Context (Parallel)
    raw_web_text = ""
    web_sources = []
    if web_func:
        # Assuming web_func is updated to return a dict as discussed
        web_data = web_func(question)
        if isinstance(web_data, dict):
            raw_web_text = web_data.get("answer", "") # or "raw_context"
            web_sources = web_data.get("urls", [])
        else:
            raw_web_text = web_data

    # 3. Merge Contexts into your RAG_TEMPLATE
    combined_context = f"--- PDF DATA ---\n{pdf_context}\n\n--- WEB DATA ---\n{raw_web_text}"
    
    # 4. Single LLM Call to Compare and Answer
    full_prompt = RAG_TEMPLATE.format(
        context=combined_context,
        question=question
    )
    
    # Using the LLM already built into your rag_chain
    response = rag_chain.combine_docs_chain.llm_chain.llm.invoke(full_prompt)
    answer = getattr(response, 'content', response)

    return {
        "answer": answer,
        "source": "hybrid",
        "sources": list(set(pdf_sources + web_sources))
    }
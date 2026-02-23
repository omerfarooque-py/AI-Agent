from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document


search = DuckDuckGoSearchRun()


def web_search(query):
    # 1. Get raw search results
    raw_results = search.run(query) 
    
    # 2. Package them as a Document so the RAG chain understands them
    web_doc = Document(
        page_content=raw_results,
        metadata={"source": "Web Search", "url": "..."}
    )
    return [web_doc]


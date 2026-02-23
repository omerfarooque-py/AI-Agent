from langchain_community.tools import DuckDuckGoSearchRun


search = DuckDuckGoSearchRun()


def web_search_tool(query: str):
    result = search.run(query)
    return {
        "answer": result,
        "source": "web"
    }


def pdf_search_tool(query: str, qa_chain):
    result = qa_chain.invoke({
        "question": query,
        "chat_history": []
    })

    answer = result.get("answer", "")

    sources = [
        doc.metadata.get("source", "Unknown")
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer": answer,
        "source": "pdf",
        "documents": sources
    }
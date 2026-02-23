from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """
You are a helpful AI assistant with access to two sources: Local PDFs and Web Search.

INSTRUCTIONS:
1. Check both the PDF and Web data below to answer the question.
2. If the sources conflict (e.g., PDF says one thing, Web says another), mention both.
3. If you find the answer, always list the source (Filename or Web URL).
4. If the answer is truly nowhere in the data, only then say "I could not find the answer."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
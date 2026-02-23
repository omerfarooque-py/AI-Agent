from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE

def process_pdf(file_path: str, original_filename: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter =  RecursiveCharacterTextSplitter(
         chunk_size = PDF_CHUNK_SIZE,
         chunk_overlap =  PDF_CHUNK_OVERLAP
    )

    splits = splitter.split_documents(docs)

    for doc in splits:
        doc.metadata["source"] = original_filename

    return splits    
import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts   import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


#checking API

if "GROQ_API_KEY" in st.secrets:
   groq_api_key = st.secrets["GROQ_API_KEY"]
else:
   st.error("please add an API key first.")
   st.stop()



# --- Page Config ---
st.set_page_config(page_title="Omer's AI Agent", page_icon="🤖")
st.title("Private AI")
st.markdown("Query your local PDFs.")

# --- Constants ---

MODEL_NAME = "llama-3.3-70b-versatile"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "./faiss_db"



if "llm" not in st.session_state:
   st.session_state.llm = ChatGroq(
      groq_api_key = groq_api_key,
      model_name = MODEL_NAME,
      temperature=0

   )

embeddings = OllamaEmbeddings(model=EMBED_MODEL)




#initializing Memory

if "memory" not in st.session_state:
   st.session_state.memory = ConversationBufferWindowMemory(
     # llm=st.session_state.llm,
      memory_key = "chat_history",
      output_key= "answer",
      return_messages = True,
      k=10

   )





def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
    vectorstore.save_local(PERSIST_DIR)
    st.session_state.vectorstore = vectorstore  # store in session
    return vectorstore



#laod existing vectors

if "vectorstore" not in st.session_state:
   if os.path.exists(PERSIST_DIR):
      st.session_state.vectorstore = FAISS.load_local(
         PERSIST_DIR, embeddings, allow_dangerous_deserialization=True

      )
   else:
      st.session_state.vectorestore = None



# --- Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Functions ---


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.info(f"Brain: {MODEL_NAME}\nEyes: {EMBED_MODEL}")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix="pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            st.spinner("analyzing document...")
            process_pdf(tmp_path)
            os.remove(tmp_path)

            st.success("knowledge base updated")
    else:
       st.warning("please upload a file first.")        

# --- Chat Interface ---

# Display chat history
if "chat_history_display" not in st.session_state:
   st.session_state.chat_history_display = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



#chain

custom_template = """You are a helpful AI. Use the context to answer the question.
If the answer isn't in the context, say you don't know.
Context: {context}
Question: {question}
Answer:"""


#QA_PROMPT = ChatPromptTemplate(template = custom_template, input_variables =["context", "question"])

QA_PROMPT = ChatPromptTemplate.from_template(custom_template)

if st.session_state.vectorstore is not None:
 qa_chain = ConversationalRetrievalChain.from_llm(
   llm = st.session_state.llm,
   retriever = st.session_state.vectorstore.as_retriever(
      search_type = "similarity",
      search_kwargs = {"k": 3}
   ),
   memory = st.session_state.memory,
   combine_docs_chain_kwargs = {"prompt" : QA_PROMPT},
   rephrase_question = True,
   return_source_documents = True
  )
else:
   qa_chain =None
   


# promt loop

if prompt := st.chat_input("Ask a question about your document..."):    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
      if qa_chain is None:
         st.error("please upload a pdf first.")    
      else:
         with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": prompt})

          
         answer = response["answer"]
         source_docs = response["source_documents"]
         
         st.sidebar.write(f"Retrieved {len(source_docs)} chunks.")
         if source_docs:
            st.sidebar.info(source_docs[0].page_content[:300])
         
         st.markdown(answer)
         st.session_state.chat_history.append({"role": "assistant", "content": answer})   

            

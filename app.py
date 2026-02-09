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
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_classic import hub
import time

#checking API

if "GROQ_API_KEY" in st.secrets:
   groq_api_key = st.secrets["GROQ_API_KEY"]
else:
   st.error("please add an API key first.")
   st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Omer's AI Agent v1.1")
st.sidebar.caption("Powered by Groq & LangChain")
# --- Page Config ---
st.set_page_config(page_title="Omer's AI Agent", page_icon="🤖")
st.markdown(""" 
# 🤖 Private AI Agent  
Secure • Local Knowledge • Groq Powered
            """)
st.markdown("---")
st.markdown(
"""
<style>
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}
</style>
""",
unsafe_allow_html=True
)




# --- Constants ---

EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "./faiss_db"

if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
    docs = st.session_state.vectorstore.docstore._dict.values()
    st.session_state.uploaded_docs = set(
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in docs
    )



if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Speed (8B)"
    st.session_state.current_model_id = "llama-3.1-8b-instant"


#switch between models

def sync_llm():
   MODEL_MAP = {
      "Speed (8B)": "llama-3.1-8b-instant",
      "Expert (70B)": "llama-3.3-70b-versatile"
   }
   selected_id = MODEL_MAP[st.session_state.model_choice]

   if "current_model_id" not in st.session_state or st.session_state.current_model_id != selected_id:
      st.session_state.current_model_id = selected_id


      st.session_state.llm = ChatGroq(
         groq_api_key = groq_api_key,
         model_name = selected_id,
         temperature  = 0,
         streaming = True

      )
      if "agent_executor" in st.session_state:
         del st.session_state["agent_executor"]


# Initialize LLM on first load
if "llm" not in st.session_state:
    MODEL_MAP = {
        "Speed (8B)": "llama-3.1-8b-instant",
        "Expert (70B)": "llama-3.3-70b-versatile"
    }
    selected_id = MODEL_MAP[st.session_state.model_choice]
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=selected_id,
        temperature=0,
        streaming=True
    )

embeddings = OllamaEmbeddings(model=EMBED_MODEL)



#initializing Memory

if "memory" not in st.session_state:
   st.session_state.memory = ConversationBufferWindowMemory(
     # llm=st.session_state.llm,
      memory_key = "chat_history",
      output_key= "output",
      return_messages = True,
      k=10

   )





def process_pdf(file_path, original_filename):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    # Add filename metadata
    for doc in splits:
        doc.metadata["source"] = original_filename
        # keep the original filename explicitly to avoid showing temp paths
        doc.metadata["original_source"] = original_filename
        # also store the local path used during processing
        doc.metadata["source_path"] = file_path

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(
            splits,
            embeddings
        )
    else:
        st.session_state.vectorstore.add_documents(splits)

    st.session_state.vectorstore.save_local(PERSIST_DIR)
    
    if "uploaded_docs" not in st.session_state:
       st.session_state.uploaded_docs = set()
    
    st.session_state.uploaded_docs.add(original_filename)




# load existing vectors


if "vectorstore" not in st.session_state:
   if os.path.exists(PERSIST_DIR):
      st.session_state.vectorstore = FAISS.load_local(
         PERSIST_DIR, embeddings, allow_dangerous_deserialization=True

      )
      # populate uploaded_docs from vectorstore metadata, prefer original_source
      try:
             docs = st.session_state.vectorstore.docstore._dict.values()
             st.session_state.uploaded_docs = set(
                  os.path.basename(doc.metadata.get("original_source") or doc.metadata.get("source", "Unknown"))
                  for doc in docs
             )
      except Exception:
             st.session_state.uploaded_docs = set()
   else:
      st.session_state.vectorstore = None



# --- Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# --- Functions ---


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.subheader("📂 Knowledge Base")
    if st.session_state.vectorstore is not None and "uploaded_docs" in st.session_state and st.session_state.uploaded_docs:
       for doc in st.session_state.uploaded_docs:
           st.caption(f"📄 {doc}")
    else:
       st.info("No documents indexed.")      
    st.selectbox(
    "Choose your Brain:",
    ["Speed (8B)", "Expert (70B)"],
    key="model_choice",
    on_change=sync_llm)
    
    uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files= True)
   
if st.sidebar.button("update Knowledge Base"):
    if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    process_pdf(tmp_path, uploaded_file.name)
                    st.session_state.uploaded_docs.add(uploaded_file.name)
                    st.write(f"✅ Processed: {uploaded_file.name}")
                    os.remove(tmp_path)
            st.success("knowledge base updated")
    else:
        st.warning("please upload a file first.") 

if st.sidebar.button("🗑️ Clear Chat History"):
     st.session_state.chat_history = []
     st.session_state.memory.clear()
     st.rerun()             

st.success(f"Active Model: {st.session_state.current_model_id}")
st.sidebar.markdown("### 🚀 Performance")
st.sidebar.caption(f"**Model:** {st.session_state.current_model_id}")       

# --- Chat Interface ---

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



#chain

custom_template = """You are a helpful AI. Use the context to answer the question.
If the answer isn't in the context, say you don't know.

EXTEREMLY IMPORTANT: If you find the answer from first tool don't use other tool.
IMPORTANT: when you have found the answer from the context or tools, you must
respond in following format:
Thought: I have information needed.
Final Answer: [Your clear and detailed answer here]

Context: {context}
Question: {question}
Answer:"""




QA_PROMPT = ChatPromptTemplate.from_template(custom_template)

def build_qa_chain():
 if st.session_state.vectorstore is None:
     return None 
 
 return ConversationalRetrievalChain.from_llm(
      llm = st.session_state.llm,
      retriever = st.session_state.vectorstore.as_retriever(
      search_type = "similarity",
      search_kwargs = {"k": 4}
   ),
   memory = st.session_state.memory,
   combine_docs_chain_kwargs = {"prompt" : QA_PROMPT},
   return_source_documents = True
  )
   


#source function.
def pdf_search_wrapper(query): 
   # Cache QA chain in session state to avoid rebuilding on every call
   if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
      st.session_state.qa_chain = build_qa_chain()
   
   qa_chain = st.session_state.qa_chain
   if qa_chain is None: 
      return "No PDF knowledge base available. Try Web Search instead."
   final_result = None 
   for chunk in qa_chain.stream({"question": query, "chat_history": []}): 
      if "answer" in chunk: 
         final_result = chunk
   if not final_result:
      return "No answer found in PDFs. Please try Web Search for current information."
   answer = final_result.get("answer", "")
   if "Final Answer:" in answer:
     answer = answer.split("Final Answer:")[-1].strip()
   elif "Thought:" in answer:
     answer = answer.split("Thought:")[0].strip()

   if not answer or answer.lower() in ["i don't know", "unknown", "no answer", ""]:
      return "PDF search did not find an answer. Try Web Search for this question."

   sources = []
   for doc in final_result.get("source_documents", []):
      # resolve source name: prefer original_source, fallback to source
      raw_src = doc.metadata.get("original_source") or doc.metadata.get("source", "Unknown PDF")
      src_basename = os.path.basename(raw_src)
      # if the basename looks like a temporary filename, try to guess a real name
      import re
      if re.match(r"^tmp", src_basename) or re.match(r"^temp", src_basename):
          # if there's exactly one uploaded doc, assume it's that
          uploaded = list(st.session_state.get("uploaded_docs", [])) if "uploaded_docs" in st.session_state else []
          if len(uploaded) == 1:
              source_name = uploaded[0]
          elif len(uploaded) > 1:
              # multiple uploaded docs: try to find one that shares page counts? fallback to first two names
              source_name = uploaded[0]
          else:
              source_name = src_basename
      else:
          source_name = src_basename
      page_num = doc.metadata.get("page", 0) + 1
      sources.append(f"📄 {source_name} (Page {page_num})")

   unique_sources = list(set(sources))
   
   if unique_sources:
         # Only store pdf sources if returning a successful answer
         try:
            st.session_state.last_pdf_sources = unique_sources
         except Exception:
                   pass
         return f"FOUND IN PDF:\n{answer}\n\nSources:\n" + "\n".join(unique_sources)
  
   return answer
#Tools

search = DuckDuckGoSearchRun()

def web_search_wrapper(query):
    """Run a web search and format output with explicit source URLs.
    Returns a string starting with 'FOUND ON WEB:' when results are present.
    """
    try:
        result = search.run(query)
    except Exception as e:
        return f"Web search error: {e}"

    # Normalize result to string
    text = result if isinstance(result, str) else str(result)

    # Simple URL extraction
    import re
    urls = re.findall(r"https?://\S+", text)
    unique_urls = []
    for u in urls:
        if u not in unique_urls:
            unique_urls.append(u)

    # store last web sources in session state for UI display
    try:
        st.session_state.last_web_sources = unique_urls
    except Exception:
        # session_state may not be available in some contexts
        pass

    if unique_urls:
        return f"FOUND ON WEB:\n{text}\n\nSources:\n" + "\n".join(unique_urls)
    # ensure we clear any previous web sources when no urls found
    try:
        st.session_state.last_web_sources = []
    except Exception:
        pass
    return f"FOUND ON WEB:\n{text}"

tools = [
   Tool(
      name="PDF_Knowledge_Base",
      func=pdf_search_wrapper, 
      description="Search uploaded PDF documents for answers. Use this FIRST for any question. Only search the web if the answer is definitely not in the PDFs."
   ),
    Tool(
        name="Web_Search",
        func=web_search_wrapper,
        description="Search the internet for current information or facts not found in PDFs. Returns FOUND ON WEB: plus Sources: URLs when available."
    )
]



prompt_template = hub.pull("hwchase17/react")

# Initialize agent if not exists or if agent_executor was deleted
if "agent_executor" not in st.session_state:
    agent = create_react_agent(
        st.session_state.llm,
        tools,
        prompt_template
    )
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="force"
    )

# promt loop

if not st.session_state.chat_history:
    st.info("Upload a PDF and start asking questions.")



if prompt := st.chat_input("Ask a question about your document..."):    
    # Clear previous sources for this new query
    st.session_state.last_pdf_sources = []
    st.session_state.last_web_sources = []
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
      if st.session_state.vectorstore is None:
         st.error("please upload a pdf first.")    
      else:
         start_time = time.time()
         response_placeholder = st.empty()
         full_response = ""
         
         try:
             # Stream shows reasoning steps in real-time
             current_step = ""
             for step in st.session_state.agent_executor.stream({"input": prompt}):
                 # Show agent's reasoning steps
                 if "actions" in step:
                     actions = step["actions"]
                     if actions:
                         action = actions[0]
                         current_step = f"🔍 Using tool: {action.tool}\n"
                         response_placeholder.markdown(current_step + "⏳ Processing...")
                 
                 # Show intermediate results
                 if "steps" in step:
                     steps = step["steps"]
                     if steps:
                         step_obj = steps[0]
                         current_step += f"\n✓ Tool result received\n"
                         response_placeholder.markdown(current_step + "⏳ Generating answer...")
                 
                 # Show final output
                 if "output" in step:
                     full_response = step["output"]
                     # Clean up step display, show final answer
                     response_placeholder.markdown(full_response)
             
             # Fallback if stream produced no output
             if not full_response:
                 result = st.session_state.agent_executor.invoke({"input": prompt})
                 full_response = result.get("output", "No response generated.")
                 response_placeholder.markdown(full_response)
         except Exception as e:
             full_response = f"Error: {str(e)}"
             response_placeholder.markdown(full_response)
         
         end_time = time.time()
         duration = round(end_time - start_time, 2)

         answer = full_response
         st.session_state.chat_history.append({"role": "assistant", "content": answer})

         # Refresh the app so the top-level chat history display shows the new messages
         try:
             st.experimental_rerun()
         except Exception:
             # If rerun isn't available in this context, continue without crashing
             pass

         # Show duration, model and last web sources (if any)
         sources_display = ""
         try:
             web_srcs = st.session_state.get("last_web_sources", [])
             pdf_srcs = st.session_state.get("last_pdf_sources", [])
             parts = []
             # shorten PDF sources (remove emoji and keep filename)
             if pdf_srcs:
                 pdf_short = []
                 for p in pdf_srcs[:3]:
                     try:
                         pdf_short.append(p.replace("📄 ", ""))
                     except Exception:
                         pdf_short.append(p)
                 parts.append("PDF: " + ", ".join(pdf_short))

             if web_srcs:
                 # shorten URLs for display (show hostname)
                 import urllib.parse
                 web_short = []
                 for u in web_srcs[:3]:
                     try:
                         pr = urllib.parse.urlparse(u)
                         host = pr.netloc or u
                         web_short.append(host)
                     except Exception:
                         web_short.append(u)
                 parts.append("Web: " + ", ".join(web_short))

             if parts:
                 sources_display = " | Sources: " + " | ".join(parts)
             else:
                 sources_display = ""
         except Exception:
             sources_display = ""

         st.caption(f"⏱️ {duration}s | 🧠 {st.session_state.current_model_id}" + sources_display)


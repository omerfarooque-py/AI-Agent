import streamlit as st
import tempfile
import os
import time


# Core imports (your engine)
from core.llm import build_llm
from core.vectorstore import (
    load_vectorstore,
    save_vectorstore,
    create_vectorstore_from_docs,
)
from core.cache import get_vectorstore
from core.pdf_processor import process_pdf
from core.rag_engine import build_rag_chain
from core.web_engine import web_search
from core.orchestrator import run_controlled_rag

#features

from features.weather import display_weather_gui

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(page_title="Omer's AI Agent", page_icon="🤖")

st.title("🤖 Private AI Agent")
st.markdown("Secure • Local Knowledge")
st.markdown("---")

# ---------------------------
# LOAD API KEY
# ---------------------------

if "GROQ_API_KEY" not in st.secrets:
    st.error("Please add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]


# ---------------------------
# SESSION STATE INIT
# ---------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if "model_type" not in st.session_state:
    st.session_state.model_type = "speed"

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = [] # Start with an empty list

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:
    st.header("Settings")

    st.selectbox(
        "Choose Model",
        ["speed", "expert"],
        key="model_type"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=True,
        key = f"pdf_uploader{st.session_state.uploader_key}"
    )
    st.markdown("---")    
    st.subheader("Uploaded files")
    if st.session_state.indexed_files:
        for fname in st.session_state.indexed_files:
          st.caption(f"✅ {fname}")
    else:
        st.info("No files indexed") 

    if st.button("Update Knowledge Base"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name

                    docs = process_pdf(tmp_path, file.name)

                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = create_vectorstore_from_docs(docs)
                    else:
                        st.session_state.vectorstore.add_documents(docs)

                    save_vectorstore(st.session_state.vectorstore)
                    st.success("saved to disk!")
                    if file.name not in st.session_state.indexed_files:
                       st.session_state.indexed_files.append(file.name)
                    os.remove(tmp_path)
            st.success("Knowledge Base Updated.")
         
        else:
            st.warning("Upload at least one PDF.")     
        st.session_state.uploader_key += 1
        st.rerun()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()



# ---------------------------
# BUILD LLM + RAG
# ---------------------------

llm = build_llm(api_key, st.session_state.model_type)

rag_chain = None
if st.session_state.vectorstore:
    rag_chain = build_rag_chain(
        llm,
        st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
    )


# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------
# CHAT INPUT
# ---------------------------

if prompt := st.chat_input("Ask a question..."):

    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        start_time = time.time()

        result = run_controlled_rag(
            question=prompt,
            rag_chain=rag_chain,
            web_func=web_search,
            chat_history=st.session_state.chat_history
        )

        answer = result["answer"]
        source_type = result["source"]
        sources_used = result.get("sources", [])

        is_weather = "weather" in prompt.lower() or "temperature" in prompt.lower()

        if is_weather:
            weather_data = {
                "temp": 15,
                "city": "Hyderabad, Pakistan",
                "condition": "Mainly Clear",
                "feels_like": 14,
                "humidity": 45,
                "wind": 9.4,
                "visibility": 10
            }
            display_weather_gui(weather_data)

            with st.expander("Show Text Summary"):
                 st.write(answer)
        else:
            st.markdown(answer)

      

        # --- SOURCE DISPLAY (Moved inside the bubble) ---
        if sources_used:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(sources_used):
                    # Handle if src is a string or a Document object
                    name = src.metadata.get("source") if hasattr(src, "metadata") else src
                    st.write(f"{i+1}. {os.path.basename(str(name))}")
        else:
            st.caption("No specific citations available.")

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        st.caption(f"⏱️ {duration}s | Source: {source_type} | Mode: {st.session_state.model_type}")

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
            
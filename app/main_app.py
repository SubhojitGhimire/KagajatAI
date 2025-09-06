import os
import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
from src.llm_interface import LLMInterface

st.set_page_config(
    page_title="KagajatAI: Chat with Your Documents",
    page_icon="📚",
    layout="wide"
)

# Streamlit's caching to load models only once
@st.cache_resource
def load_llm_interface():
    try:
        return LLMInterface()
    except Exception as e:
        st.error(f"Failed to initialise LLM Interface: {e}")
        st.stop()
@st.cache_resource
def load_rag_pipeline():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialise RAG Pipeline: {e}")
        st.stop()

# Initialising session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm KagajatAI. Upload a document and I'll help you understand it.")
    ]
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = os.path.exists('data/vector_store')

def main():
    with st.spinner("Initialising AI components... Please wait! This may take a moment."):
        llm_interface = load_llm_interface()
        rag_pipeline = load_rag_pipeline()
    
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            source_dir = rag_pipeline.config['data']['source_documents_path']
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(source_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)
            st.success(f"Saved files: {', '.join(saved_files)}")

        if st.button("Ingest Documents"):
            if not os.listdir(rag_pipeline.config['data']['source_documents_path']):
                st.warning("No documents found in the source directory to ingest.")
            else:
                with st.spinner("Ingesting documents into vector store. Please wait..."):
                    rag_pipeline.ingest_documents()
                st.session_state.vector_store_loaded = True
                st.success("Documents ingested successfully!")

        st.info(f"Using LLM Provider: {llm_interface.llm_provider.upper()}")
        if not st.session_state.vector_store_loaded:
            st.warning("Vector store is empty. Please upload and ingest documents.")

    # Main Chat Interface
    st.title("KagajatAI: Chat with Your Documents")
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        if not st.session_state.vector_store_loaded:
            st.error("Cannot process query. Please upload and ingest documents first.")
            return

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            with st.spinner("Searching documents..."):
                retriever = rag_pipeline.get_retriever()
                retrieved_docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            rag_chain = llm_interface.create_rag_chain()
            with st.spinner("Generating answer..."):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in rag_chain.stream({"context": context, "question": user_query}):
                    content_chunk = chunk.content if hasattr(chunk, 'content') else chunk
                    full_response += content_chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            final_answer = full_response.split("ASSISTANT:")[-1].strip()
            st.session_state.chat_history.append(AIMessage(content=final_answer))
            
            # Sources for transparency.
            with st.expander("View Retrieved Context", expanded=False):
                st.info("The AI's answer was generated based on the following document excerpts:")
                for i, doc in enumerate(retrieved_docs):
                    source_name = os.path.basename(doc.metadata.get('source', 'N/A'))
                    page_num = doc.metadata.get('page', 'N/A')
                    st.write(f"**Source {i+1}:** *{source_name} (Page {page_num})*")
                    st.write(f"> {doc.page_content.strip()}")
                    st.divider()

if __name__ == "__main__":
    main()

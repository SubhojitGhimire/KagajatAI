import os
import yaml
import torch
from typing import List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.data_processing import load_documents_from_directory, split_documents_into_chunks

class RAGPipeline:
    def __init__(self, config_path: str = 'config/config.yaml'):
        print("Initialising RAG Pipeline...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embedding_model = self._init_embeddings()
        self.vector_store = self._init_vector_store()
        print("RAG Pipeline Initialised Successfully.")

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        embedding_config = self.config['embedding']
        model_name = embedding_config['model_name']
        device = embedding_config.get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available. Falling back to CPU for embeddings.")
            device = 'cpu'
        print(f"Initialising embedding model '{model_name}' on device '{device}'.")
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': self.config['embedding']['normalize']}

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def _init_vector_store(self) -> Chroma:
        vector_store_path = self.config['data']['vector_store_path']
        print(f"Loading vector store from: {vector_store_path}")
        return Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embedding_model
        )

    def ingest_documents(self):
        source_dir = self.config['data']['source_documents_path']
        # 1. Load documents
        documents = load_documents_from_directory(source_dir)
        if not documents:
            print("No new documents to ingest.")
            return
        # 2. Split documents
        chunks = split_documents_into_chunks(documents)
        if not chunks:
            print("Document splitting resulted in no chunks. Ingestion halted.")
            return
        # 3. Add to vector store
        print(f"Ingesting {len(chunks)} chunks into the vector store...")
        self.vector_store.add_documents(chunks)
        print("Document ingestion complete.")

    def get_retriever(self, search_k: int = 4):
        print(f"Creating a retriever to fetch top {search_k} results.")
        return self.vector_store.as_retriever(search_kwargs={'k': search_k})

# Sample test for Chroma retriever
if __name__ == '__main__':
    print("Running RAG pipeline ingestion script...")
    rag_pipeline = RAGPipeline()
    rag_pipeline.ingest_documents()

    print("\n--- Performing a test retrieval ---")
    retriever = rag_pipeline.get_retriever()
    sample_query = "What are the key highlights?"
    try:
        retrieved_docs = retriever.invoke(sample_query)
        print(f"Query: '{sample_query}'")
        print(f"Retrieved {len(retrieved_docs)} documents.")
        if retrieved_docs:
            print("\n--- Sample Retrieved Chunk ---")
            print(retrieved_docs[0].page_content)
            print("\n--- Metadata ---")
            print(retrieved_docs[0].metadata)
            print("--------------------------")
        else:
            print("No documents retrieved. The vector store might be empty or the query irrelevant.")
    except Exception as e:
        print(f"An error occurred during test retrieval: {e}")
        print("This might happen if the vector store is empty. Run the ingestion process first.")
    print("\n---------------------------------")
    print("Ingestion script finished.")


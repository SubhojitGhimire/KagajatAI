import os
import yaml
from typing import List

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_config(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_documents_from_directory(directory_path: str) -> List[Document]:
    print(f"Loading documents from: {directory_path}")
    if not os.path.isdir(directory_path):
        raise Exception(f"Directory not found at path: {directory_path}")
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print(f"Warning: No documents loaded from {directory_path}. Make sure there are PDF files in the directory.")
        return []
        
    print(f"Successfully loaded {len(documents)} document(s).")
    return documents

def split_documents_into_chunks(documents: List[Document]) -> List[Document]:
    config = load_config()
    chunk_size = config['data']['chunk_size']
    chunk_overlap = config['data']['chunk_overlap']

    print(f"Splitting {len(documents)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split documents into {len(chunks)} chunks.")
    return chunks

# Sample test for deep document understanding and chunking
if __name__ == '__main__':
    config = load_config()
    source_dir = config['data']['source_documents_path']
    loaded_docs = load_documents_from_directory(source_dir)
    if loaded_docs:
        document_chunks = split_documents_into_chunks(loaded_docs)
        print("\n--- Sample Chunk ---")
        print(document_chunks[0].page_content)
        print("\n--- Metadata ---")
        print(document_chunks[0].metadata)
        print("\n--------------------")


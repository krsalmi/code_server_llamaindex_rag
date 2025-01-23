import os
import json
from typing import List, Dict

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document
)

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from dotenv import load_dotenv

STORAGE_DIR = "./storage"
JSONLINES_FILE_PATH = "./icd_codes_rag.jsonl"
EMBEDDING_DIMENSION = 1536  # Embedding dimension for "text-embedding-ada-002"

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def prepare_documents(doc_dicts):
    docs = []
    for doc in doc_dicts:
        text = json.dumps(doc)
        metadata = {"code": doc.get("code"), "description": doc.get("description"),
                    "synonyms": doc.get("synonyms"),
                    "parent_code": doc.get("parent_code"), 
                    "parent_description": doc.get("parent_description")}
        docs.append(Document(text=text, extra_info=metadata))
    return docs

def prepare_index(documents, storage_dir):
    d = EMBEDDING_DIMENSION
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    os.makedirs(storage_dir, exist_ok=True)  # Ensure the directory exists
    index.storage_context.persist(persist_dir=storage_dir)
    return index

def initialize_index():
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise ValueError("OpenAI API key not found. Please set it in .env file.")

    try:
        jsonlines = load_jsonl(JSONLINES_FILE_PATH)
        print(f"Loaded {len(jsonlines)} documents from {JSONLINES_FILE_PATH}")
        documents = prepare_documents(jsonlines)
        prepare_index(documents, STORAGE_DIR)
        print(f"Index prepared and saved to '{STORAGE_DIR}'.")
    except FileNotFoundError:
        print(f"File not found: {JSONLINES_FILE_PATH}")

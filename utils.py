import json, re
from llama_index.core import (
    load_index_from_storage,
    StorageContext
)
from llama_index.vector_stores.faiss import FaissVectorStore

STORAGE_DIR = "./storage"


def prepare_retriever():
    vector_store = FaissVectorStore.from_persist_dir(STORAGE_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=STORAGE_DIR
    )
    index = load_index_from_storage(storage_context=storage_context)

    # Set up the retriever
    retriever = index.as_retriever(similarity_top_k=1)
    return retriever

def process_multiline_string(multiline_str):
    """
    Processes a multiline string by splitting it into a list of non-empty, trimmed strings.

    Parameters:
    - multiline_str (str): The input multiline string.

    Returns:
    - list of str: A list containing non-empty, trimmed lines from the input string.
    """
    # Compile a regex pattern to match lines containing "ICD", "10", and "code" (answers can start often with "ICD-10 codes:")
    pattern = re.compile(r'icd.*10.*code', re.IGNORECASE)

    # Split the string into lines based on the newline character
    lines = multiline_str.split('\n')

    # Use list comprehension to strip whitespace and exclude empty lines
    processed_lines = [line.strip() for line in lines if line.strip() and not pattern.search(line)]

    return processed_lines


def make_json_objects(documents):
  json_docs = []
  for doc in documents:
    json_doc = json.loads(doc)
    json_docs.append(json_doc)
  return json_docs

def filter_unique_parent_codes(json_docs):
    """
    Filters the input list of JSON objects, retaining only the first occurrence
    of each unique parent_code.

    Parameters:
    - json_docs (list of dict): The list of JSON objects to filter.

    Returns:
    - list of dict: A new list containing only the first occurrence of each parent_code.
    """
    seen_parent_codes = set()
    filtered_docs = []

    for doc in json_docs:
        parent_code = doc.get('parent_code')
        if parent_code not in seen_parent_codes:
            seen_parent_codes.add(parent_code)
            filtered_docs.append(doc)
    return filtered_docs

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DOCUMENT_PATH = BASE_DIR / "rag" / "documents" / "churn_knowledge.txt"

import faiss
import numpy as np
from rag.document import load_document
from rag.chunker import create_chunks
from rag.embeddings import create_embeddings
from rag.vector import create_vector_store
from rag.retriever import search_documents


def test_knowledge_base_exists():
    assert DOCUMENT_PATH.exists()

def test_knowledge_base_not_empty():
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert len(document.strip()) > 0

def test_knowledge_base_contains_churn_information():
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert "churn" in document.lower()
    
    
def test_load_document():
    document=load_document()
    assert isinstance(document, str)
    assert len(document.strip())>0     
    

def test_create_chunks():
    document = load_document()
    chunks = create_chunks(document)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk.strip()) > 0 for chunk in chunks)     
    
    


def test_create_embeddings():
    chunks = ["Customer churn means a customer stops using a service."]    
    embeddings = create_embeddings(chunks)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0


def test_create_vector_store():
    embeddings = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]], dtype="float32")
    index = faiss.IndexFlatL2(3)
    index.add(embeddings)
    assert index.ntotal == 2 
    assert index.d == 3    
    
def test_search_documents():
    document = load_document()
    chunks = create_chunks(document)
    question = "Which contract has higher churn risk?"
    results = search_documents(question, chunks, top_k=2)
    assert len(results) > 0
    assert all(isinstance(result, str) for result in results)    
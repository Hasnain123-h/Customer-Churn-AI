from pathlib import Path

import faiss
import numpy as np
from rag.document import load_document
from rag.chunker import create_chunks
from rag.embeddings import create_embeddings


BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_store.index"


def create_vector_store(embeddings):    
    embeddings = np.array(embeddings).astype("float32") 
    dimension = embeddings.shape[1]
    print("Embedding Dimension:", dimension)   
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    print("Vectors Added:", index.ntotal)

  
    faiss.write_index(
        index,
        str(VECTOR_PATH)
    )

    print("Vector Store Path:", VECTOR_PATH)

    return index


if __name__ == "__main__":

    print("Creating Vector Store...")
   
    document = load_document()
    print("Document Loaded.")    
    chunks = create_chunks(document)
    print("Total Chunks:", len(chunks))    
    embeddings = create_embeddings(chunks)
    print("Embeddings Created.")    
    index = create_vector_store(embeddings)    
    print("\nVector Store Created Successfully.")
    print("Number of Vectors:", index.ntotal)

    if VECTOR_PATH.exists():
        print("File Created Successfully:")
        print(VECTOR_PATH)
    else:
        print("ERROR: Vector Store File Was Not Created.")
from pathlib import Path
import faiss
import numpy as np
from rag.document import load_document
from rag.chunker import create_chunks
from rag.embeddings import create_embeddings

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_store.index"


def load_vector_store():
    index = faiss.read_index(
        str(VECTOR_PATH)
    )

    return index


def search_documents(question, chunks, top_k=2):    
    question_embedding = create_embeddings(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    index = load_vector_store()

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for index_value in indices[0]:
        if index_value != -1:
            results.append(
                chunks[index_value]
            )

    return results


if __name__ == "__main__":

    document = load_document()
    chunks = create_chunks(document)
    question = "Which contract has higher churn risk?"
    results = search_documents(
        question,
        chunks
    )

    print("Question:")
    print(question)
    print("\nRelevant Documents:")

    for result in results:
        print("\n---")
        print(result)
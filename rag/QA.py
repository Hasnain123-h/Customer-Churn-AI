from retriever import search_documents
from document import load_document
from chunker import create_chunks
from generator import generate_answer


def answer_question(question, customer_info=None):
    document = load_document()
    chunks = create_chunks(document)
    relevant_documents = search_documents(
        question,
        chunks,
        top_k=2
    )

    context = "\n\n".join(
        relevant_documents
    )

    answer = generate_answer(
        question,
        context,
        customer_info
    )

    return answer
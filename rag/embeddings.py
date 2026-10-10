from sentence_transformers import SentenceTransformer


# Embedding model load करना
model = SentenceTransformer("all-MiniLM-L6-v2")
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    from document import load_document
    from chunker import create_chunks
    document = load_document()
    chunks = create_chunks(document)
    embeddings = create_embeddings(chunks)
    print("Number of Chunks:", len(chunks))
    print("Embedding Shape:", embeddings.shape)
    print("\nFirst Embedding:")
    print(embeddings[0])


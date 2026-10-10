from rag.document import load_document

def create_chunks(document, chunk_size =300, overlap=50):
    chunks=[]
    step = chunk_size - overlap
    for i in range(0, len(document), step):
        chunk=document[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    document = load_document()
    chunks=create_chunks(document)
    print("chunks:", len(chunks))
    print(chunks[0])    
        
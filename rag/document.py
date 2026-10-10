from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "documents" / "churn_knowledge.txt"

def load_document():
    document = DOCUMENT_PATH.read_text(
        encoding="utf-8"
    )
    return document

if __name__ == "__main__":
    document = load_document()
    print(document)

# chunk_pdfs.py

import os
import PyPDF2
from typing import List, Tuple
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    tokens = enc.encode(text)
    chunks: List[str] = []
    start = 0
    text_len = len(tokens)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)
        chunks.append(chunk_str)
        if end == text_len:
            break
        start += chunk_size - overlap
    return chunks

def chunk_pdfs_with_ids(pdf_folder: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for file in os.listdir(pdf_folder):
        if not file.endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, file)
        print(f"Extracting {file}")
        raw_text = extract_text_from_pdf(path)
        print(f"Chunking {file}")
        chunks = chunk_text(raw_text, chunk_size=512, overlap=50)
        # inside chunk_pdfs_with_ids(...)
        doc_id = os.path.splitext(file)[0]  # ensure string, not tuple
        for c in chunks:
            out.append((c, doc_id))  # (text:str, doc_id:str)

    return out

if __name__ == "__main__":
    pdf_folder = "/Users/sadiakhantumpa/Desktop/Thesis MSc/pdfs"
    pairs = chunk_pdfs_with_ids(pdf_folder)
    print(f"Total chunks created: {len(pairs)}")

    # Write TSV: doc_id \t text
    # TSV write section at bottom
    with open("text_chunks.tsv", "w", encoding="utf-8") as f:
        for text, doc_id in pairs:  # unpack in same order as appended
            safe_doc = doc_id.replace("\n", " ")
            safe_txt = text.replace("\n", " ")
            print(type(pairs), type(pairs), type(pairs)) # Debugging line
            f.write(f"{safe_doc}\t{safe_txt}\n")

    print("Chunks saved to text_chunks.tsv")

# Terminal commands
# pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers torch
import re

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for i, row in df.iterrows():
        prompt_text = str(row.get("prompt", ""))
        choices = []

        for choice in ["A", "B", "C", "D", "E"]:
            if choice in row and pd.notna(row[choice]):
                choices.append(f"{choice}: {row[choice]}")

        text = prompt_text.strip()
        if choices:
            # Put each choice on its own line so semantic chunking can preserve
            # answer-option boundaries instead of splitting inside a choice.
            text = f"{text}\n\nChoices:\n" + "\n".join(choices)

        metadata = {
            "source": f"row_{int(row.get('id', i))}",
            "answer": row.get("answer", ""),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def simple_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def build_splitter(chunk_size: int, chunk_overlap: int, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=simple_token_count,
    )


def fixed_chunk_documents(documents, chunk_size: int, chunk_overlap: int, label: str):
    splitter = build_splitter(chunk_size, chunk_overlap, separators=[""])
    chunks = splitter.split_documents(documents)
    print(f"{label}: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, chunks={len(chunks)}")
    return chunks


def split_semantic_sections(text: str):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    sections = []

    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if lines and lines[0] == "Choices:":
            choice_lines = lines[1:]
            if not choice_lines:
                sections.append("Choices:")
                continue
            sections.append("Choices:\n" + choice_lines[0])
            sections.extend(choice_lines[1:])
        else:
            sections.append(paragraph)

    return sections


def retain_overlap_sections(sections, overlap_tokens: int):
    retained = []
    token_total = 0

    for section in reversed(sections):
        section_tokens = simple_token_count(section)
        if retained and token_total + section_tokens > overlap_tokens:
            break
        retained.insert(0, section)
        token_total += section_tokens
        if token_total >= overlap_tokens:
            break

    return retained


def semantic_chunk_documents(documents, chunk_size: int, chunk_overlap: int, label: str):
    splitter = build_splitter(chunk_size, chunk_overlap)
    output_documents = []

    for document in documents:
        sections = split_semantic_sections(document.page_content)
        if not sections:
            continue

        buffer = []
        buffer_tokens = 0

        def flush_buffer():
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            chunk_text = "\n\n".join(buffer).strip()
            if chunk_text:
                output_documents.append(
                    Document(page_content=chunk_text, metadata=dict(document.metadata))
                )
            buffer = retain_overlap_sections(buffer, chunk_overlap)
            buffer_tokens = simple_token_count("\n\n".join(buffer)) if buffer else 0

        for section in sections:
            section_tokens = simple_token_count(section)
            if section_tokens > chunk_size:
                flush_buffer()
                large_docs = splitter.split_documents(
                    [Document(page_content=section, metadata=dict(document.metadata))]
                )
                output_documents.extend(large_docs)
                continue

            if buffer and buffer_tokens + section_tokens > chunk_size:
                flush_buffer()

            buffer.append(section)
            buffer_tokens += section_tokens

        flush_buffer()

    print(f"{label}: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, chunks={len(output_documents)}")
    return output_documents


def load_and_chunk_data(chunk_size: int, method: str = "A"):
    """
    method: 'A' for fixed chunking, 'B' for semantic/recursive chunking, 'both' for both methods
    """
    csv_path = "data/train.csv"
    documents = load_documents(csv_path)
    chunk_overlap = max(0, int(chunk_size * 0.1))

    print(f"Loaded {len(documents)} rows from {csv_path}\n")

    if method in ["A", "both"]:
        # Method A: Fixed-Size Chunking
        chunks_a = fixed_chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            label="Method A (Fixed-Size Chunking)",
        )
        if method == "A":
            return chunks_a

    if method in ["B", "both"]:
        # Method B: keep prompt/choice boundaries when possible, then recursively
        # split only oversized sections.
        chunks_b = semantic_chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            label="Method B (Semantic/Recursive Chunking)",
        )
        if method == "B":
            return chunks_b

    if method == "both":
        return chunks_a, chunks_b


if __name__ == "__main__":
    # Change to 512 or 1024, and method to 'A', 'B', or 'both' as needed
    chunk_size = 1024
    method = "both"  # or "A" or "B"
    
    result = load_and_chunk_data(chunk_size, method)
    if method == "both":
        chunks_a, chunks_b = result
        print(f"\nComparison: Method A={len(chunks_a)} chunks, Method B={len(chunks_b)} chunks")
    else:
        print(f"\nTotal chunks: {len(result)}")

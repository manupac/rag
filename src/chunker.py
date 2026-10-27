"""Load the parsed rulebook Markdown, clean it, and split it into chunks."""

import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import CHUNK_OVERLAP, CHUNK_SIZE, MD_PATH

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def clean_markdown(text: str) -> str:
    """Remove parsing artifacts that only add noise to the embeddings."""
    text = re.sub(r"<!-- page \d+ -->\n?", "", text)  # page markers
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", text)  # image references
    text = re.sub(r"</?mark>", "", text)  # highlight tags
    text = text.replace("~~", "")  # strikethrough artifacts (it still leaves some blank spaces in words e.g. "Rule s")
    text = re.sub(r"^#+[ \t]*\**[ \t]*\d+[ \t]*\**[ \t]*$\n?", "", text, flags=re.MULTILINE)  # page numbers parsed as headers
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse leftover blank lines
    return text


def create_chunks(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split the rulebook into chunks. Defaults come from config; the ablation script passes others."""
    markdown_document = clean_markdown(MD_PATH.read_text(encoding="utf-8"))

    # split on headers, so each piece belongs to one section.
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=True)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # split long sections into chunks of at most chunk_size characters
    # separators are tried in order; "" (split anywhere) must be last
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "- ", " ", ""],
    )
    final_chunks = text_splitter.split_documents(md_header_splits)

    # prepend the header path to every chunk, so sub-chunks of a long
    # section still say which section they come from
    for chunk in final_chunks:
        header_path = " > ".join(
            chunk.metadata[name].replace("*", "").strip() for _, name in HEADERS_TO_SPLIT_ON if name in chunk.metadata
        )
        if header_path:
            chunk.page_content = f"{header_path}\n\n{chunk.page_content}"

    return final_chunks


def main():
    chunks = create_chunks()
    sizes = [len(c.page_content) for c in chunks]
    print(f"{len(chunks)} chunks, average {sum(sizes) // len(sizes)} chars, max {max(sizes)}")
    print("\n--- First chunk ---")
    print(chunks[0].page_content)


if __name__ == "__main__":
    main()
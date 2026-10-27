"""Convert the rulebook PDF to Markdown with both marker-pdf and PyMuPDF, then merge them page by page.

For each page, marker's version is used, unless it has less than 85% of the words in
PyMuPDF's version, in which case PyMuPDF's version is used. The full marker and PyMuPDF
versions are also saved, for manual inspection."""

import re
import subprocess

import pymupdf4llm

from config import MD_PATH, PARSED_DIR, PDF_PATH

MARKER_MD = PARSED_DIR / "marker" / PDF_PATH.stem / f"{PDF_PATH.stem}.md"

MIN_COVERAGE = 0.85  # marker must contain this fraction of PyMuPDF's words
MIN_WORDS = 20  # below this, a page is treated as an image page


def parse_pymupdf() -> list[str]:
    pymupdf4llm.use_layout(True)
    return [c["text"] for c in pymupdf4llm.to_markdown(str(PDF_PATH), page_chunks=True)]


def parse_marker(n_pages: int) -> list[str]:
    if not MARKER_MD.exists():  # marker is slow, so run it only once
        subprocess.run(
            [
                "marker_single",
                "--disable_ocr",
                "--output_dir",
                str(PARSED_DIR / "marker"),
                "--paginate_output",
                str(PDF_PATH),
            ],
            check=True,
        )
    # Pages are separated by lines like: {3}------------------------------------
    parts = re.split(r"\n*\{(\d+)\}-{20,}\n*", MARKER_MD.read_text(encoding="utf-8"))
    pages = {int(num): text.strip() for num, text in zip(parts[1::2], parts[2::2])}
    if not pages:
        raise SystemExit(f"No page separators found in {MARKER_MD}; check its format.")
    return [pages.get(i, "") for i in range(n_pages)]  # pages marker dropped -> ""


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def marker_is_incomplete(marker: str, pymupdf: str) -> bool:
    reference = words(pymupdf)
    if len(reference) < MIN_WORDS:
        return False
    return len(reference & words(marker)) / len(reference) < MIN_COVERAGE


def save(pages: list[str], name: str) -> None:
    text = "\n\n".join(f"<!-- page {i} -->\n{page}" for i, page in enumerate(pages, start=1))
    (PARSED_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    pymupdf = parse_pymupdf()
    marker = parse_marker(len(pymupdf))

    final = [p if marker_is_incomplete(m, p) else m for m, p in zip(marker, pymupdf)]
    replaced = [i for i, (m, p) in enumerate(zip(marker, pymupdf), start=1) if marker_is_incomplete(m, p)]

    save(marker, "rulebook_marker.md")
    save(pymupdf, "rulebook_pymupdf.md")
    save(final, MD_PATH.name)
    print(f"{len(replaced)}/{len(final)} pages taken from PyMuPDF: {replaced}")


if __name__ == "__main__":
    main()
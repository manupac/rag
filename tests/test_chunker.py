import chunker
from chunker import clean_markdown, create_chunks
from config import CHUNK_SIZE


# ---------- clean_markdown ----------


def test_removes_page_markers():
    assert clean_markdown("<!-- page 3 -->\nHello") == "Hello"


def test_removes_image_references():
    assert clean_markdown("![diagram](img/pic.png)\ntext") == "text"


def test_removes_mark_tags():
    assert clean_markdown("<mark>important</mark>") == "important"


def test_removes_strikethrough_markers():
    assert clean_markdown("~~Rule~~s") == "Rules"


def test_collapses_blank_lines():
    assert clean_markdown("a\n\n\n\nb") == "a\n\nb"


def test_plain_text_is_unchanged():
    assert clean_markdown("Just a normal sentence.") == "Just a normal sentence."


# ---------- create_chunks ----------


def write_rulebook(tmp_path, monkeypatch, text):
    path = tmp_path / "rulebook.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(chunker, "MD_PATH", path)


def test_chunks_start_with_their_header_path(tmp_path, monkeypatch):
    write_rulebook(
        tmp_path,
        monkeypatch,
        "# Shooting Phase\n\nIntro text.\n\n## Hit Roll\n\nRoll a D6.\n",
    )
    contents = [c.page_content for c in create_chunks()]

    assert any(c.startswith("Shooting Phase\n\nIntro text") for c in contents)
    assert any(c.startswith("Shooting Phase > Hit Roll\n\nRoll a D6") for c in contents)


def test_bold_markers_are_removed_from_header_path(tmp_path, monkeypatch):
    write_rulebook(tmp_path, monkeypatch, "# **Bold Title**\n\nSome text.\n")
    chunks = create_chunks()

    assert chunks[0].page_content.startswith("Bold Title\n\n")
    assert "*" not in chunks[0].page_content.split("\n\n")[0]


def test_long_sections_are_split_and_keep_their_header(tmp_path, monkeypatch):
    long_body = "word " * 600  # about 3000 characters, far over CHUNK_SIZE
    write_rulebook(tmp_path, monkeypatch, f"# Big\n\n{long_body}\n")
    chunks = create_chunks()

    assert len(chunks) > 1
    header_length = len("Big\n\n")
    for chunk in chunks:
        assert chunk.page_content.startswith("Big\n\n")
        assert len(chunk.page_content) <= CHUNK_SIZE + header_length


def test_page_markers_never_reach_the_chunks(tmp_path, monkeypatch):
    write_rulebook(tmp_path, monkeypatch, "<!-- page 1 -->\n# Title\n\nBody.\n")
    for chunk in create_chunks():
        assert "<!-- page" not in chunk.page_content

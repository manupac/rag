import sys
import types

import pytest

# parser.py imports pymupdf4llm at the top. The tests never parse a real PDF,
# so a stand-in module is enough (and keeps the tests fast).
sys.modules.setdefault("pymupdf4llm", types.ModuleType("pymupdf4llm"))

import parser as rulebook_parser  # noqa: E402


def numbered_words(n):
    return " ".join(f"w{i}" for i in range(n))


# ---------- words / marker_is_incomplete ----------


def test_words_lowercases_and_deduplicates():
    assert rulebook_parser.words("Hello, hello WORLD!") == {"hello", "world"}


def test_short_reference_pages_are_never_incomplete():
    # fewer than MIN_WORDS (20) words in PyMuPDF's page: treated as an image page
    assert rulebook_parser.marker_is_incomplete("", numbered_words(10)) is False


def test_marker_with_full_coverage_is_complete():
    text = numbered_words(30)
    assert rulebook_parser.marker_is_incomplete(text, text) is False


def test_marker_with_low_coverage_is_incomplete():
    assert rulebook_parser.marker_is_incomplete(numbered_words(10), numbered_words(30)) is True


def test_coverage_threshold_is_85_percent():
    pymupdf = numbered_words(100)
    assert rulebook_parser.marker_is_incomplete(numbered_words(85), pymupdf) is False
    assert rulebook_parser.marker_is_incomplete(numbered_words(84), pymupdf) is True


def test_empty_marker_page_is_incomplete_when_pymupdf_has_text():
    assert rulebook_parser.marker_is_incomplete("", numbered_words(30)) is True


# ---------- parse_pymupdf ----------


def test_parse_pymupdf_returns_text_of_each_page(monkeypatch):
    monkeypatch.setattr(rulebook_parser.pymupdf4llm, "use_layout", lambda flag: None, raising=False)
    monkeypatch.setattr(
        rulebook_parser.pymupdf4llm,
        "to_markdown",
        lambda path, page_chunks: [{"text": "page one"}, {"text": "page two"}],
        raising=False,
    )
    assert rulebook_parser.parse_pymupdf() == ["page one", "page two"]


# ---------- parse_marker ----------

SEPARATOR = "-" * 48


def write_marker_output(path, pages: dict[int, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(f"{{{num}}}{SEPARATOR}\n\n{body}" for num, body in pages.items())
    path.write_text(text, encoding="utf-8")


def test_parse_marker_splits_pages_by_separator(tmp_path, monkeypatch):
    marker_md = tmp_path / "marker.md"
    write_marker_output(marker_md, {0: "Page zero text", 1: "Page one text"})
    monkeypatch.setattr(rulebook_parser, "MARKER_MD", marker_md)

    assert rulebook_parser.parse_marker(2) == ["Page zero text", "Page one text"]


def test_parse_marker_fills_dropped_pages_with_empty_string(tmp_path, monkeypatch):
    marker_md = tmp_path / "marker.md"
    write_marker_output(marker_md, {0: "first", 2: "third"})  # marker dropped page 1
    monkeypatch.setattr(rulebook_parser, "MARKER_MD", marker_md)

    assert rulebook_parser.parse_marker(3) == ["first", "", "third"]


def test_parse_marker_exits_when_no_separators_found(tmp_path, monkeypatch):
    marker_md = tmp_path / "marker.md"
    marker_md.write_text("no separators here", encoding="utf-8")
    monkeypatch.setattr(rulebook_parser, "MARKER_MD", marker_md)

    with pytest.raises(SystemExit):
        rulebook_parser.parse_marker(1)


def test_parse_marker_does_not_rerun_marker_if_output_exists(tmp_path, monkeypatch):
    marker_md = tmp_path / "marker.md"
    write_marker_output(marker_md, {0: "cached"})
    monkeypatch.setattr(rulebook_parser, "MARKER_MD", marker_md)

    def fail(*args, **kwargs):
        raise AssertionError("marker should not run when its output is cached")

    monkeypatch.setattr(rulebook_parser.subprocess, "run", fail)
    assert rulebook_parser.parse_marker(1) == ["cached"]


def test_parse_marker_runs_marker_when_output_is_missing(tmp_path, monkeypatch):
    marker_md = tmp_path / "marker" / "out.md"
    monkeypatch.setattr(rulebook_parser, "MARKER_MD", marker_md)
    commands = []

    def fake_run(command, check):
        commands.append(command)
        write_marker_output(marker_md, {0: "fresh"})

    monkeypatch.setattr(rulebook_parser.subprocess, "run", fake_run)

    assert rulebook_parser.parse_marker(1) == ["fresh"]
    assert len(commands) == 1
    assert commands[0][0] == "marker_single"


# ---------- save / main ----------


def test_save_tags_each_page_with_its_number(tmp_path, monkeypatch):
    monkeypatch.setattr(rulebook_parser, "PARSED_DIR", tmp_path)
    rulebook_parser.save(["alpha", "beta"], "out.md")

    assert (tmp_path / "out.md").read_text(encoding="utf-8") == (
        "<!-- page 1 -->\nalpha\n\n<!-- page 2 -->\nbeta"
    )


def test_main_writes_three_files_and_prefers_pymupdf_for_incomplete_pages(tmp_path, monkeypatch):
    long_page = numbered_words(30)
    monkeypatch.setattr(rulebook_parser, "PARSED_DIR", tmp_path)
    monkeypatch.setattr(rulebook_parser, "MD_PATH", tmp_path / "rulebook_final.md")
    monkeypatch.setattr(rulebook_parser, "parse_pymupdf", lambda: [long_page, "short pymupdf page"])
    monkeypatch.setattr(rulebook_parser, "parse_marker", lambda n: ["w0 w1", "marker page two"])

    rulebook_parser.main()

    final = (tmp_path / "rulebook_final.md").read_text(encoding="utf-8")
    assert long_page in final  # page 1: marker was incomplete, PyMuPDF used
    assert "marker page two" in final  # page 2: PyMuPDF page too short to judge, marker kept
    assert "short pymupdf page" not in final
    assert (tmp_path / "rulebook_marker.md").exists()
    assert (tmp_path / "rulebook_pymupdf.md").exists()

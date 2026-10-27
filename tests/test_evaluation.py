import json
from types import SimpleNamespace

import pytest

import evaluation
from evaluation import evaluate_item, format_rate, judge, rate, summarize


class FakeBot:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def ask(self, question):
        if self.error:
            raise self.error
        return self.response


def make_response(route="RULEBOOK", sources=(), sql=None, answer="An answer."):
    return SimpleNamespace(route=route, sources=list(sources), sql=sql, answer=answer)


def doc(text):
    return SimpleNamespace(page_content=text)


RULEBOOK_ITEM = {
    "id": "q1",
    "question": "How does charging work?",
    "expected_route": "RULEBOOK",
    "reference_answer": "Roll 2D6.",
    "source_text": "Charge phase\n\nRoll 2D6.",
}
DATABASE_ITEM = {
    "id": "q2",
    "question": "Toughness of a Carnifex?",
    "expected_route": "DATABASE",
    "reference_answer": "9",
    "source_table": "units",
}


def always_correct(question, reference, answer):
    return True, "CORRECT matches"


# ---------- judge ----------


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("CORRECT - it matches the reference", True),
        ("correct", True),
        ("**CORRECT** the facts match", True),
        ("INCORRECT - key facts are missing", False),
        ("incorrect: contradicts the reference", False),
    ],
)
def test_judge_reads_the_verdict(monkeypatch, reply, expected):
    monkeypatch.setattr(evaluation, "ask_llm", lambda system, user: reply)
    correct, text = judge("q", "ref", "ans")
    assert correct is expected
    assert text == reply.strip()


def test_judge_prompt_contains_all_three_parts(monkeypatch):
    seen = []
    monkeypatch.setattr(evaluation, "ask_llm", lambda system, user: seen.append(user) or "CORRECT")
    judge("the question", "the reference", "the answer")

    assert all(part in seen[0] for part in ("the question", "the reference", "the answer"))


# ---------- evaluate_item ----------


def test_rulebook_question_routed_right_and_chunk_retrieved():
    bot = FakeBot(make_response(sources=[doc("other"), doc(RULEBOOK_ITEM["source_text"])]))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is True
    assert result["sql_ok"] is None
    assert result["correct"] is True
    assert bot.reset_calls == 1  # history is cleared before each question


def test_rulebook_question_where_the_chunk_was_not_retrieved():
    bot = FakeBot(make_response(sources=[doc("unrelated chunk")]))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is False


def test_wrong_route_leaves_retrieval_and_sql_metrics_empty():
    sql = SimpleNamespace(query="SELECT 1", error=None, rows=[(1,)])
    bot = FakeBot(make_response(route="DATABASE", sql=sql))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is False
    assert result["retrieval_hit"] is None
    assert result["sql_ok"] is None
    assert result["sql_query"] == "SELECT 1"


def test_database_question_with_working_query():
    sql = SimpleNamespace(query="SELECT toughness", error=None, rows=[("9",)])
    result = evaluate_item(FakeBot(make_response(route="DATABASE", sql=sql)), DATABASE_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["sql_ok"] is True
    assert result["retrieval_hit"] is None


@pytest.mark.parametrize(
    "sql",
    [
        SimpleNamespace(query="bad", error="no such table", rows=[]),
        SimpleNamespace(query="SELECT 1 WHERE 0", error=None, rows=[]),
    ],
)
def test_database_question_with_failing_or_empty_query(sql):
    result = evaluate_item(FakeBot(make_response(route="DATABASE", sql=sql)), DATABASE_ITEM, always_correct)
    assert result["sql_ok"] is False


def test_error_while_asking_is_recorded_not_raised():
    result = evaluate_item(FakeBot(error=ValueError("boom")), RULEBOOK_ITEM, always_correct)

    assert result["error"] == "boom"
    assert result["correct"] is False
    assert result["route_ok"] is False


def test_judge_failure_is_not_counted_as_wrong():
    def broken_judge(question, reference, answer):
        raise RuntimeError("judge down")

    result = evaluate_item(FakeBot(make_response()), RULEBOOK_ITEM, broken_judge)

    assert result["correct"] is None
    assert "judge down" in result["judge_reply"]


# ---------- rate / summarize / format_rate ----------


def test_rate_ignores_none_values():
    assert rate([True, False, None, True]) == {"passed": 2, "total": 3, "rate": 2 / 3}


def test_rate_of_nothing_is_none():
    assert rate([None, None]) == {"passed": 0, "total": 0, "rate": None}
    assert rate([])["rate"] is None


def test_format_rate():
    assert format_rate({"passed": 3, "total": 4, "rate": 0.75}) == "3/4 (75%)"
    assert format_rate({"passed": 0, "total": 0, "rate": None}) == "n/a"


def test_summarize_counts_each_metric_over_its_own_questions():
    results = [
        {"route_ok": True, "retrieval_hit": True, "sql_ok": None, "correct": True, "error": None},
        {"route_ok": True, "retrieval_hit": False, "sql_ok": None, "correct": False, "error": None},
        {"route_ok": False, "retrieval_hit": None, "sql_ok": None, "correct": False, "error": None},
        {"route_ok": True, "retrieval_hit": None, "sql_ok": True, "correct": None, "error": None},
        {"route_ok": False, "retrieval_hit": None, "sql_ok": None, "correct": False, "error": "boom"},
    ]
    summary = summarize(results)

    assert summary["n_questions"] == 5
    assert summary["n_errors"] == 1
    assert summary["routing_accuracy"]["passed"] == 3 and summary["routing_accuracy"]["total"] == 5
    assert summary["retrieval_hit_rate"]["passed"] == 1 and summary["retrieval_hit_rate"]["total"] == 2
    assert summary["sql_success_rate"]["passed"] == 1 and summary["sql_success_rate"]["total"] == 1
    assert summary["answer_correctness"]["passed"] == 1 and summary["answer_correctness"]["total"] == 4


# ---------- main ----------


def test_main_runs_every_question_and_saves_results(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "results.json"
    sql = SimpleNamespace(query="SELECT 1", error=None, rows=[("9",)])
    responses = {
        RULEBOOK_ITEM["question"]: make_response(sources=[doc(RULEBOOK_ITEM["source_text"])]),
        DATABASE_ITEM["question"]: make_response(route="DATABASE", sql=sql),
    }

    class StubBot:
        def reset(self):
            pass

        def ask(self, question):
            return responses[question]

    sleeps = []
    monkeypatch.setattr(evaluation, "load_questions", lambda: [RULEBOOK_ITEM, DATABASE_ITEM])
    monkeypatch.setattr(evaluation, "Chatbot", StubBot)
    monkeypatch.setattr(evaluation, "judge", always_correct)
    monkeypatch.setattr(evaluation, "EVAL_RESULTS_PATH", results_path)
    monkeypatch.setattr(evaluation.time, "sleep", sleeps.append)

    evaluation.main()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["summary"]["n_questions"] == 2
    assert saved["summary"]["routing_accuracy"]["rate"] == 1.0
    assert saved["summary"]["answer_correctness"]["rate"] == 1.0
    assert [r["id"] for r in saved["results"]] == ["q1", "q2"]
    assert sleeps == [evaluation.EVAL_PAUSE_SECONDS]  # a pause between questions, none after the last
    assert "Evaluation summary" in capsys.readouterr().out

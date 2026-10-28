import json
from types import SimpleNamespace

import pytest

import evaluation
from evaluation import evaluate_item, format_rate, grade, is_finished, judge, rate, summarize
from llm import DailyQuotaError


class FakeBot:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.reset_calls = 0
        self.asked = []

    def reset(self):
        self.reset_calls += 1

    def ask(self, question):
        self.asked.append(question)
        if self.error:
            raise self.error
        return self.response


def make_response(route="RULEBOOK", sources=(), sql=None, answer="An answer.", standalone_question="q"):
    return SimpleNamespace(
        route=route, sources=list(sources), sql=sql, answer=answer, standalone_question=standalone_question
    )


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
BOTH_ITEM = {
    "id": "b1",
    "question": "Can a Carnifex shoot in Engagement Range?",
    "expected_route": "BOTH",
    "reference_answer": "Yes, with -1 to Hit.",
    "source_text": "Big Guns Never Tire\n\nMonsters can shoot.",
}


def always_correct(question, reference, answer):
    return True, "CORRECT matches"


def broken_judge(question, reference, answer):
    raise RuntimeError("judge down")


# ---------- judge ----------


def use_judge_llm(monkeypatch, fake_llm, reply):
    fake = fake_llm([reply])
    monkeypatch.setattr(evaluation, "get_judge_llm", lambda: fake)
    return fake


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
def test_judge_reads_the_verdict(monkeypatch, fake_llm, reply, expected):
    use_judge_llm(monkeypatch, fake_llm, reply)
    correct, text = judge("q", "ref", "ans")
    assert correct is expected
    assert text == reply.strip()


def test_judge_prompt_contains_all_three_parts(monkeypatch, fake_llm):
    fake = use_judge_llm(monkeypatch, fake_llm, "CORRECT")
    judge("the question", "the reference", "the answer")

    system_message, human_message = fake.calls[0]
    assert system_message.content == evaluation.JUDGE_SYSTEM
    assert all(part in human_message.content for part in ("the question", "the reference", "the answer"))


# ---------- grade ----------


def test_grade_fills_in_correct_and_judge_reply():
    result = {"question": "q", "reference_answer": "r", "answer": "a"}
    grade(result, always_correct)

    assert result["correct"] is True
    assert result["judge_reply"] == "CORRECT matches"


def test_grade_records_a_judge_failure_without_counting_it():
    result = grade({"question": "q", "reference_answer": "r", "answer": "a"}, broken_judge)

    assert result["correct"] is None
    assert result["judge_reply"] == "judge failed: judge down"


def test_grade_marks_a_daily_quota_stop():
    def out_of_quota(question, reference, answer):
        raise DailyQuotaError("PerDay")

    result = grade({"question": "q", "reference_answer": "r", "answer": "a"}, out_of_quota)

    assert result["correct"] is None
    assert result["judge_reply"] == evaluation.QUOTA_MSG


def test_grade_uses_the_real_judge_by_default(monkeypatch):
    monkeypatch.setattr(evaluation, "judge", lambda q, r, a: (False, "INCORRECT nope"))
    result = grade({"question": "q", "reference_answer": "r", "answer": "a"})

    assert result["correct"] is False


# ---------- evaluate_item ----------


def test_rulebook_question_routed_right_and_chunk_retrieved():
    bot = FakeBot(make_response(sources=[doc("other"), doc(RULEBOOK_ITEM["source_text"])]))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is True
    assert result["sql_ok"] is None
    assert result["correct"] is True
    assert bot.reset_calls == 1  # history is cleared before each question
    assert RULEBOOK_ITEM["source_text"] in result["retrieved_contexts"]

def test_rulebook_question_where_the_chunk_was_not_retrieved():
    bot = FakeBot(make_response(sources=[doc("unrelated chunk")]))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is False


def test_question_without_source_text_has_no_retrieval_metric():
    item = {k: v for k, v in RULEBOOK_ITEM.items() if k != "source_text"}
    result = evaluate_item(FakeBot(make_response(sources=[doc("x")])), item, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is None


def test_wrong_route_leaves_retrieval_and_sql_metrics_empty():
    sql = SimpleNamespace(query="SELECT 1", error=None, rows=[(1,)], columns=["x"], truncated=False)    
    bot = FakeBot(make_response(route="DATABASE", sql=sql))
    result = evaluate_item(bot, RULEBOOK_ITEM, always_correct)

    assert result["route_ok"] is False
    assert result["retrieval_hit"] is None
    assert result["sql_ok"] is None
    assert result["sql_query"] == "SELECT 1"


def test_database_question_with_working_query():
    sql = SimpleNamespace(query="SELECT toughness", error=None, rows=[("9",)], columns=["x"], truncated=False)
    result = evaluate_item(FakeBot(make_response(route="DATABASE", sql=sql)), DATABASE_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["sql_ok"] is True
    assert result["retrieval_hit"] is None


def test_both_question_scores_retrieval_and_sql():
    sql = SimpleNamespace(query="SELECT keyword", error=None, rows=[("Monster",)], columns=["x"], truncated=False)
    bot = FakeBot(make_response(route="BOTH", sources=[doc(BOTH_ITEM["source_text"])], sql=sql))
    result = evaluate_item(bot, BOTH_ITEM, always_correct)

    assert result["route_ok"] is True
    assert result["retrieval_hit"] is True
    assert result["sql_ok"] is True
    assert result["correct"] is True


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


def test_history_is_asked_first_in_the_same_conversation():
    item = {**RULEBOOK_ITEM, "history": ["first question"]}
    bot = FakeBot(make_response(sources=[doc(RULEBOOK_ITEM["source_text"])]))
    result = evaluate_item(bot, item, always_correct)

    assert bot.asked == ["first question", RULEBOOK_ITEM["question"]]
    assert bot.reset_calls == 1
    assert result["history"] == ["first question"]


def test_error_while_asking_is_recorded_not_raised():
    result = evaluate_item(FakeBot(error=ValueError("boom")), RULEBOOK_ITEM, always_correct)

    assert result["error"] == "boom"
    assert result["correct"] is None
    assert result["route_ok"] is False


def test_daily_quota_while_asking_is_raised_so_the_run_can_stop():
    with pytest.raises(DailyQuotaError):
        evaluate_item(FakeBot(error=DailyQuotaError("PerDay")), RULEBOOK_ITEM, always_correct)


def test_judge_failure_is_not_counted_as_wrong():
    result = evaluate_item(FakeBot(make_response()), RULEBOOK_ITEM, broken_judge)

    assert result["correct"] is None
    assert "judge down" in result["judge_reply"]
    assert result["answer"] == "An answer."  # the answer is kept, so it is only judged again later


# ---------- is_finished ----------


def test_is_finished_only_for_answered_and_judged_results():
    assert is_finished(None) is False
    assert is_finished({"error": "boom", "correct": None}) is False
    assert is_finished({"error": None, "correct": None}) is False  # answered, not judged
    assert is_finished({"error": None, "correct": False}) is True
    assert is_finished({"error": None, "correct": True}) is True


# ---------- rate / summarize / format_rate ----------


def test_rate_ignores_none_values():
    assert rate([True, False, None, True]) == {"passed": 2, "total": 3, "rate": 2 / 3}


def test_rate_of_nothing_is_none():
    assert rate([None, None]) == {"passed": 0, "total": 0, "rate": None}
    assert rate([])["rate"] is None


def test_format_rate():
    assert format_rate({"passed": 3, "total": 4, "rate": 0.75}) == "3/4 (75%)"
    assert format_rate({"passed": 0, "total": 0, "rate": None}) == "n/a"


def make_summary_row(expected_route, route_ok, retrieval_hit=None, sql_ok=None, correct=None, error=None, history=()):
    return {
        "expected_route": expected_route,
        "history": list(history),
        "route_ok": route_ok,
        "retrieval_hit": retrieval_hit,
        "sql_ok": sql_ok,
        "correct": correct,
        "error": error,
    }


def test_summarize_counts_each_metric_over_its_own_questions():
    results = [
        make_summary_row("RULEBOOK", True, retrieval_hit=True, correct=True),
        make_summary_row("RULEBOOK", True, retrieval_hit=False, correct=False),
        make_summary_row("RULEBOOK", False, correct=False),
        make_summary_row("DATABASE", True, sql_ok=True),
        make_summary_row("BOTH", False, correct=False, error="boom"),
    ]
    summary = summarize(results)

    assert summary["n_questions"] == 5
    assert summary["n_errors"] == 1
    assert summary["n_unjudged"] == 1  # the database row: answered, no verdict, no error
    assert summary["routing_accuracy"]["passed"] == 3 and summary["routing_accuracy"]["total"] == 5
    assert summary["routing_by_expected"]["RULEBOOK"] == {"passed": 2, "total": 3, "rate": 2 / 3}
    assert summary["routing_by_expected"]["DATABASE"] == {"passed": 1, "total": 1, "rate": 1.0}
    assert summary["routing_by_expected"]["BOTH"] == {"passed": 0, "total": 1, "rate": 0.0}
    assert summary["retrieval_hit_rate"]["passed"] == 1 and summary["retrieval_hit_rate"]["total"] == 2
    assert summary["sql_success_rate"]["passed"] == 1 and summary["sql_success_rate"]["total"] == 1
    assert summary["answer_correctness"]["passed"] == 1 and summary["answer_correctness"]["total"] == 4
    assert summary["multi_turn_correctness"]["rate"] is None  # no question had a history


def test_summarize_multi_turn_only_counts_questions_with_history():
    results = [
        make_summary_row("RULEBOOK", True, correct=True, history=["earlier"]),
        make_summary_row("RULEBOOK", True, correct=False, history=["earlier"]),
        make_summary_row("RULEBOOK", True, correct=True),
    ]
    summary = summarize(results)

    assert summary["multi_turn_correctness"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary["answer_correctness"]["total"] == 3


# ---------- main ----------


class StubBot:
    """Answers from `responses` (by question) and records what it was asked, across all instances."""

    responses = {}
    asked = []

    def reset(self):
        pass

    def ask(self, question):
        StubBot.asked.append(question)
        if isinstance(StubBot.responses, Exception):
            raise StubBot.responses
        return StubBot.responses[question]


@pytest.fixture
def run_main(tmp_path, monkeypatch, sleeps):
    """Prepares evaluation.main() with fake questions, a stub bot and a temporary results file."""
    results_path = tmp_path / "results.json"
    sql = SimpleNamespace(query="SELECT 1", error=None, rows=[("9",)], columns=["x"], truncated=False)    
    StubBot.asked = []
    StubBot.responses = {
        RULEBOOK_ITEM["question"]: make_response(sources=[doc(RULEBOOK_ITEM["source_text"])]),
        DATABASE_ITEM["question"]: make_response(route="DATABASE", sql=sql),
    }

    monkeypatch.setattr(evaluation, "load_questions", lambda: [RULEBOOK_ITEM, DATABASE_ITEM])
    monkeypatch.setattr(evaluation, "load_manual_questions", lambda: [])
    monkeypatch.setattr(evaluation, "Chatbot", StubBot)
    monkeypatch.setattr(evaluation, "judge", always_correct)
    monkeypatch.setattr(evaluation, "EVAL_RESULTS_PATH", results_path)

    def run():
        evaluation.main()
        return json.loads(results_path.read_text(encoding="utf-8"))

    run.results_path = results_path
    return run


def write_saved(path, results):
    path.write_text(json.dumps({"summary": {}, "results": results}), encoding="utf-8")


def test_main_runs_every_question_and_saves_results(run_main, sleeps, capsys):
    saved = run_main()

    assert saved["summary"]["n_questions"] == 2
    assert saved["summary"]["routing_accuracy"]["rate"] == 1.0
    assert saved["summary"]["answer_correctness"]["rate"] == 1.0
    assert [r["id"] for r in saved["results"]] == ["q1", "q2"]
    assert sleeps == [evaluation.EVAL_PAUSE_SECONDS] * 2  # a pause after each question
    assert "Evaluation summary" in capsys.readouterr().out


def test_main_skips_questions_that_are_already_finished(run_main):
    finished = evaluate_item(FakeBot(make_response(sources=[doc(RULEBOOK_ITEM["source_text"])])), RULEBOOK_ITEM, always_correct)
    write_saved(run_main.results_path, [finished])

    saved = run_main()

    assert StubBot.asked == [DATABASE_ITEM["question"]]  # q1 was not asked again
    assert [r["id"] for r in saved["results"]] == ["q1", "q2"]


def test_main_only_judges_answers_that_were_not_judged_yet(run_main):
    unjudged = evaluate_item(FakeBot(make_response(sources=[doc("x")])), RULEBOOK_ITEM, broken_judge)
    assert unjudged["correct"] is None and unjudged["answer"] is not None
    write_saved(run_main.results_path, [unjudged])

    saved = run_main()

    assert StubBot.asked == [DATABASE_ITEM["question"]]  # q1 was not answered again
    q1 = saved["results"][0]
    assert q1["correct"] is True
    assert q1["answer"] == "An answer."


def test_main_redoes_questions_that_failed(run_main):
    failed = evaluate_item(FakeBot(error=ValueError("boom")), RULEBOOK_ITEM, always_correct)
    write_saved(run_main.results_path, [failed])

    saved = run_main()

    assert RULEBOOK_ITEM["question"] in StubBot.asked
    assert saved["results"][0]["error"] is None


def test_main_stops_cleanly_when_the_bot_hits_the_daily_quota(run_main, capsys, sleeps):
    StubBot.responses = DailyQuotaError("PerDay")

    saved = run_main()

    assert saved["results"] == []
    assert "Daily quota used up. 2 questions still to do" in capsys.readouterr().out
    assert sleeps == []


def test_main_stops_when_the_judge_hits_the_daily_quota(run_main, monkeypatch, sleeps, capsys):
    def out_of_quota(question, reference, answer):
        raise DailyQuotaError("PerDay")

    monkeypatch.setattr(evaluation, "judge", out_of_quota)

    saved = run_main()

    assert [r["id"] for r in saved["results"]] == ["q1"]  # stopped after the first question
    assert saved["results"][0]["answer"] == "An answer."  # the answer is kept for the next run
    assert saved["results"][0]["judge_reply"] == evaluation.QUOTA_MSG
    assert sleeps == []
    assert "Daily quota used up" in capsys.readouterr().out
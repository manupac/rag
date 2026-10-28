"""Run the evaluation: send every dataset question through the chatbot and measure how it did.

Metrics:
- routing accuracy: did the router pick the expected channel (RULEBOOK / DATABASE)?
- retrieval hit rate: for generated rulebook questions that were routed correctly, was the
  chunk the question came from among the retrieved chunks?
- SQL success rate: for database questions that were routed correctly, did the generated
  query run without error and return rows?
- answer correctness: an LLM judge (a different model than the bot) compares the bot's
  answer with the reference answer.

Questions come from eval/questions.json (generated) and eval/manual_questions.json
(hand-written, optional). A question can have a "history": a list of earlier questions
asked first in the same conversation, to test multi-turn follow-ups.

Results are saved to eval/results.json after every question. If the daily API quota runs
out, the run stops cleanly; running 'eval' again later resumes where it stopped:
finished questions are skipped, answered-but-unjudged ones are only judged, and failed
ones are redone. Delete eval/results.json to start from scratch (do this after changing
any setting that affects the answers, e.g. TOP_K).
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from chatbot import Chatbot
from config import EVAL_PAUSE_SECONDS, EVAL_RESULTS_PATH
from eval_dataset import load_manual_questions, load_questions
from llm import DailyQuotaError, call_with_retry, get_judge_llm, text_of
from router import BOTH, DATABASE, RULEBOOK
from sql_retriever import format_result

RULES_ROUTES = {RULEBOOK, BOTH}
DATA_ROUTES = {DATABASE, BOTH}

QUOTA_MSG = "judge skipped: daily quota used up"

JUDGE_SYSTEM = """You grade a chatbot's answer against a reference answer.

Reply CORRECT if the chatbot's answer contains the key facts of the reference answer
and does not contradict it. Extra detail is fine.
Reply INCORRECT if key facts are missing, if it contradicts the reference, or if it
says it could not find the information.

Exception: if the reference answer says the information is not available, reply CORRECT
only if the chatbot also says it could not find it, and INCORRECT if it gives an answer.

Start your reply with exactly one word, CORRECT or INCORRECT, then a short reason."""

JUDGE_USER = """Question: {question}

Reference answer: {reference}

Chatbot's answer: {answer}"""


def judge(question: str, reference: str, answer: str) -> tuple[bool, str]:
    """Ask the judge model whether the answer matches the reference. Returns (correct, reply)."""
    messages = [
        SystemMessage(JUDGE_SYSTEM),
        HumanMessage(JUDGE_USER.format(question=question, reference=reference, answer=answer)),
    ]
    reply = text_of(call_with_retry(get_judge_llm().invoke, messages)).strip()
    verdict = reply.lstrip("*`\"' ").upper()
    return verdict.startswith("CORRECT"), reply


def grade(result: dict, judge_fn=None) -> dict:
    """Judge result['answer'] and fill in 'correct' and 'judge_reply'.

    Kept separate from answering, so an answer that was already generated is never
    regenerated just because the judge failed.
    """
    judge_fn = judge_fn or judge
    try:
        result["correct"], result["judge_reply"] = judge_fn(
            result["question"], result["reference_answer"], result["answer"]
        )
    except DailyQuotaError:
        result["correct"] = None
        result["judge_reply"] = QUOTA_MSG  # main() sees this and stops the run
    except Exception as e:  # noqa: BLE001 - a judge failure is recorded, not fatal
        result["correct"] = None  # not counted
        result["judge_reply"] = f"judge failed: {e}"
    return result


def evaluate_item(bot, item: dict, judge_fn=None) -> dict:
    """Run one question (after its history, if any) through the bot and score it.

    Raises DailyQuotaError if the bot's model hit its daily limit: the question is
    then left undone and redone in the next run.
    """
    result = {
        "id": item["id"],
        "question": item["question"],
        "history": item.get("history", []),
        "expected_route": item["expected_route"],
        "reference_answer": item["reference_answer"],
        "standalone_question": None,  # how the bot rewrote the question
        "route": None,
        "route_ok": False,
        "retrieval_hit": None,  # only for generated rulebook questions routed correctly
        "sql_ok": None,  # only for database questions routed correctly
        "sql_query": None,
        "answer": None,
        "correct": None,
        "judge_reply": None,
        "error": None,
        "retrieved_contexts": []
    }

    bot.reset()  # every test starts a new conversation
    try:
        for previous in item.get("history", []):  # earlier turns, not scored
            call_with_retry(bot.ask, previous)
        response = call_with_retry(bot.ask, item["question"])
    except DailyQuotaError:
        raise  # not a failure of this question: main() stops the run
    except Exception as e:  # noqa: BLE001 - one failing question must not stop the run
        result.update(error=str(e))  # counted in n_errors, not as a wrong answer
        return result

    result["standalone_question"] = response.standalone_question
    result["route"] = response.route
    result["route_ok"] = response.route == item["expected_route"]
    result["answer"] = response.answer
    result["retrieved_contexts"] = [doc.page_content for doc in response.sources] # will help RAGAS later on
    if response.sql is not None:
        result["retrieved_contexts"].append(format_result(response.sql))

    # hand-written questions have no source chunk, so retrieval can't be checked for them
    if result["route_ok"] and item["expected_route"] in RULES_ROUTES and item.get("source_text"):
        result["retrieval_hit"] = any(doc.page_content == item["source_text"] for doc in response.sources)
    if response.sql is not None:
        result["sql_query"] = response.sql.query
    if result["route_ok"] and item["expected_route"] in DATA_ROUTES:
        result["sql_ok"] = response.sql is not None and response.sql.error is None and bool(response.sql.rows)

    return grade(result, judge_fn)


def rate(values: list) -> dict:
    """Pass rate over the values that are not None."""
    counted = [v for v in values if v is not None]
    passed = sum(1 for v in counted if v)
    return {"passed": passed, "total": len(counted), "rate": passed / len(counted) if counted else None}


def summarize(results: list[dict]) -> dict:
    return {
        "n_questions": len(results),
        "n_errors": sum(1 for r in results if r["error"]),
        "n_unjudged": sum(1 for r in results if r["correct"] is None and not r["error"]),
        "routing_accuracy": rate([r["route_ok"] for r in results]),
        "routing_by_expected": {
            r: rate([x["route_ok"] for x in results if x["expected_route"] == r]) for r in (RULEBOOK, DATABASE, BOTH)
        },
        "retrieval_hit_rate": rate([r["retrieval_hit"] for r in results]),
        "sql_success_rate": rate([r["sql_ok"] for r in results]),
        "answer_correctness": rate([r["correct"] for r in results]),
        "multi_turn_correctness": rate([r["correct"] for r in results if r["history"]]),
    }


def format_rate(stat: dict) -> str:
    if stat["rate"] is None:
        return "n/a"
    return f"{stat['passed']}/{stat['total']} ({stat['rate']:.0%})"


def mark(value) -> str:
    return "-" if value is None else ("yes" if value else "NO")


def print_item(i: int, total: int, result: dict) -> None:
    print(
        f"[{i}/{total}] {result['id']}  expected {result['expected_route']}, got {result['route']}  "
        f"| retrieval: {mark(result['retrieval_hit'])}  sql: {mark(result['sql_ok'])}  "
        f"correct: {mark(result['correct'])}"
    )
    if result["history"]:
        print(f"        rewritten as: {result['standalone_question']}")
    if result["error"]:
        print(f"        error: {result['error']}")


def print_summary(summary: dict) -> None:
    print("\n=== Evaluation summary ===")
    print(f"Questions:          {summary['n_questions']} ({summary['n_errors']} with errors)")
    print(f"Routing accuracy:   {format_rate(summary['routing_accuracy'])}")
    print(f"Retrieval hit rate: {format_rate(summary['retrieval_hit_rate'])}  (generated rulebook questions)")
    print(f"SQL success rate:   {format_rate(summary['sql_success_rate'])}  (database questions routed correctly)")
    print(f"Answer correctness: {format_rate(summary['answer_correctness'])}  (judged by the LLM)")
    print(f"  of which multi-turn: {format_rate(summary['multi_turn_correctness'])}")
    if summary["n_unjudged"]:
        print(f"WARNING: {summary['n_unjudged']} answers are not judged yet; the rates above exclude them.")


def is_finished(result: dict | None) -> bool:
    """Answered and judged: nothing left to do for this question."""
    return bool(result) and not result["error"] and result["correct"] is not None


def load_saved() -> dict:
    """Results from earlier runs, by question id."""
    if not EVAL_RESULTS_PATH.exists():
        return {}
    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["results"]}


def save(questions: list[dict], saved: dict) -> list[dict]:
    """Write the results (in question order) and return them."""
    results = [saved[q["id"]] for q in questions if q["id"] in saved]
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(
        json.dumps({"summary": summarize(results), "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def main():
    questions = load_questions() + load_manual_questions()
    saved = load_saved()
    bot = Chatbot()
    stopped = False

    for i, item in enumerate(questions, start=1):
        old = saved.get(item["id"])
        if is_finished(old):
            continue  # finished in an earlier run

        try:
            if old and not old["error"] and old["answer"] is not None:
                result = grade(old)  # answered before: only the judging is missing
            else:
                result = evaluate_item(bot, item)
        except DailyQuotaError:
            stopped = True
            break

        saved[item["id"]] = result
        save(questions, saved)  # after every question, so nothing is lost
        print_item(i, len(questions), result)

        if result["judge_reply"] == QUOTA_MSG:
            stopped = True
            break
        time.sleep(EVAL_PAUSE_SECONDS)  # stay under the free-tier per-minute limit

    results = save(questions, saved)
    print_summary(summarize(results))
    if stopped:
        remaining = sum(1 for q in questions if not is_finished(saved.get(q["id"])))
        print(f"\nDaily quota used up. {remaining} questions still to do: run 'eval' again tomorrow.")
    print(f"\nFull results (answers, SQL, judge replies) saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
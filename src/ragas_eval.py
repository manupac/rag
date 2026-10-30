"""Second opinion on the evaluation: score the saved results with RAGAS.

Reads eval/results.json (so the bot is not called again) and computes four RAGAS metrics
per question: faithfulness, context recall, context precision and factual correctness.
Scores are saved after every metric, so the run can stop at the daily quota and resume
the next day: only the missing scores are computed.
Run it in the eval environment:  pixi run -e eval python main.py ragas [ids...]
"""

import json
import math
import time

from config import EVAL_PAUSE_SECONDS, EVAL_RESULTS_PATH, RAGAS_METRICS, RAGAS_MODEL, RAGAS_RESULTS_PATH, RAGAS_ROUTES
from llm import DailyQuotaError, get_llm

def usable(result: dict) -> bool:
    """Answered without error, with contexts saved."""
    return bool(result.get("answer")) and not result["error"] and bool(result.get("retrieved_contexts"))


def to_sample_fields(result: dict) -> dict:
    return {
        # the rewritten question, so follow-ups like "and its save?" make sense on their own
        "user_input": result["standalone_question"] or result["question"],
        "response": result["answer"],
        "retrieved_contexts": result["retrieved_contexts"],
        "reference": result["reference_answer"],
    }


def make_metrics() -> dict:
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(get_llm(RAGAS_MODEL), run_config=RunConfig(max_workers=1, max_retries=6, timeout=180))
    available = {
        "faithfulness": Faithfulness(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
    }
    return {name: available[name] for name in RAGAS_METRICS}


def call_metric(metric, fields: dict) -> float | None:
    from ragas import SingleTurnSample

    try:
        value = metric.single_turn_score(SingleTurnSample(**fields))
    except Exception as e:  # noqa: BLE001
        if "PerDay" in repr(e):
            raise DailyQuotaError(str(e)) from e
        raise
    return None if value is None or math.isnan(value) else float(value)


def mean(values: list) -> dict:
    counted = [v for v in values if v is not None]
    return {"mean": sum(counted) / len(counted) if counted else None, "n": len(counted)}


def summarize(scored: dict, results: list[dict]) -> dict:
    route = {r["id"]: r["expected_route"] for r in results}
    summary = {}
    for name in RAGAS_METRICS:
        summary[name] = {
            "overall": mean([row.get(name) for row in scored.values()]),
            **{
                r: mean([row.get(name) for qid, row in scored.items() if route.get(qid) == r])
                for r in ("RULEBOOK", "DATABASE", "BOTH")
            },
        }
    return summary


def disagreements(scored: dict, results: list[dict]) -> list[str]:
    """Questions where RAGAS factual correctness and our own judge disagree."""
    lines = []
    for r in results:
        fc = scored.get(r["id"], {}).get("factual_correctness")
        if fc is not None and r["correct"] is not None and r["correct"] != (fc >= 0.5):
            lines.append(f"{r['id']}: own judge {'CORRECT' if r['correct'] else 'INCORRECT'}, RAGAS factual correctness {fc:.2f}")
    return lines


def load_scored() -> dict:
    if not RAGAS_RESULTS_PATH.exists():
        return {}
    return json.loads(RAGAS_RESULTS_PATH.read_text(encoding="utf-8"))["scores"]


def save(scored: dict, results: list[dict]) -> None:
    RAGAS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summarize(scored, results), "scores": scored}
    RAGAS_RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(ids: list[str] | None = None):
    if not EVAL_RESULTS_PATH.exists():
        raise SystemExit(f"{EVAL_RESULTS_PATH} not found. Run 'python main.py eval' first.")
    results = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    selected = [r for r in results if usable(r) and r["expected_route"] in RAGAS_ROUTES and (not ids or r["id"] in ids)]
    if not selected:
        raise SystemExit("No result with an answer and saved contexts. Run 'eval' again (see the README).")

    scored, metrics, stopped = load_scored(), make_metrics(), False
    for result in selected:
        row = scored.setdefault(result["id"], {})
        fields = to_sample_fields(result)
        for name, metric in metrics.items():
            if row.get(name) is not None:
                continue  # already scored in an earlier run
            try:
                row[name] = call_metric(metric, fields)
            except DailyQuotaError:
                stopped = True
                break
            except Exception as e:  # noqa: BLE001 - left empty, retried in the next run
                print(f"  {result['id']} {name} failed: {e}")
                row[name] = None
            save(scored, results)
            print(f"{result['id']:>4} {name:<20} {row[name]}")
            time.sleep(EVAL_PAUSE_SECONDS)
        if stopped:
            break

    save(scored, results)
    for name, stat in summarize(scored, results).items():
        o = stat["overall"]
        print(f"{name:<20} mean {o['mean'] if o['mean'] is None else round(o['mean'], 2)} over {o['n']} questions")
    for line in disagreements(scored, results):
        print("DISAGREEMENT", line)
    if stopped:
        print("\nDaily quota used up: run 'ragas' again tomorrow, it resumes where it stopped.")
    print(f"\nSaved to {RAGAS_RESULTS_PATH}")
"""LongMemEval benchmark runner for memdio.

Usage:
    python -m benchmarks.longmemeval.run                          # all models
    python -m benchmarks.longmemeval.run --model openai/gpt-4o    # single model
    python -m benchmarks.longmemeval.run --resume <run_id>        # resume
    python -m benchmarks.longmemeval.run --limit 10               # quick test with 10 questions
    python -m benchmarks.longmemeval.run --workers 8              # parallel LLM calls
"""

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from benchmarks.config import ANSWER_MODELS, CHECKPOINTS_DIR
from benchmarks.longmemeval.answer import generate_answer, get_client
from benchmarks.longmemeval.download import load_dataset
from benchmarks.longmemeval.evaluate import evaluate_single, get_judge_client
from benchmarks.longmemeval.ingest import cleanup_question_db, ingest_question
from benchmarks.longmemeval.report import print_report, save_results
from benchmarks.longmemeval.search import format_context, hybrid_search


def save_checkpoint(run_id: str, model: str, completed: dict, results: list[dict]):
    """Save checkpoint for resume capability."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"{run_id}_{model.replace('/', '_')}.json")
    with open(path, "w") as f:
        json.dump({"completed": completed, "results": results}, f)
    return path


def load_checkpoint(run_id: str, model: str) -> tuple[dict, list[dict]]:
    """Load checkpoint if exists."""
    path = os.path.join(CHECKPOINTS_DIR, f"{run_id}_{model.replace('/', '_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["completed"], data["results"]
    return {}, []


def process_question(question: dict, model: str, answer_client, judge_client) -> dict:
    """Process a single question: ingest -> search -> answer -> evaluate.

    Designed to run in a thread pool. Each question gets its own isolated DB.
    """
    qid = question["question_id"]
    qtype = question.get("question_type", "unknown")

    try:
        storage, db_dir = ingest_question(question)
    except Exception as e:
        return {
            "question_id": qid, "question_type": qtype,
            "is_abstention": "_abs" in qid, "label": False,
            "error": f"ingest: {e}",
        }

    try:
        search_results = hybrid_search(storage, question["question"])
        context = format_context(search_results, query=question["question"])

        hypothesis = generate_answer(
            answer_client, model,
            question["question"], context,
            question_date=question.get("question_date", ""),
        )

        eval_result = evaluate_single(
            judge_client,
            question_id=qid,
            question_type=qtype,
            question=question["question"],
            reference_answer=question["answer"],
            hypothesis=hypothesis,
        )
        eval_result["hypothesis"] = hypothesis
        eval_result["num_memories_found"] = len(search_results)
        return eval_result

    except Exception as e:
        return {
            "question_id": qid, "question_type": qtype,
            "is_abstention": "_abs" in qid, "label": False,
            "error": str(e),
        }
    finally:
        storage.close()
        cleanup_question_db(db_dir)


def run_benchmark(model: str, run_id: str, limit: int | None = None, workers: int = 8):
    """Run full benchmark pipeline with parallel execution."""
    print(f"\n{'=' * 60}")
    print(f"  LongMemEval Benchmark — {model}")
    print(f"  Run ID: {run_id} | Workers: {workers}")
    print(f"{'=' * 60}\n")

    dataset = load_dataset()
    if limit:
        dataset = dataset[:limit]
    print(f"Loaded {len(dataset)} questions")

    completed, results = load_checkpoint(run_id, model)
    if completed:
        print(f"Resuming from checkpoint: {len(completed)} already completed")

    # Filter out already completed
    pending = [(i, q) for i, q in enumerate(dataset) if q["question_id"] not in completed]
    print(f"Pending: {len(pending)} questions")

    if not pending:
        print("Nothing to do.")
        print_report(results, model=model)
        return results

    answer_client = get_client()
    judge_client = get_judge_client()

    start_time = time.time()
    done_count = len(completed)
    total = len(dataset)

    # Process in batches to avoid overwhelming CPU/memory
    batch_size = workers
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_q = {}
            for idx, question in batch:
                future = executor.submit(
                    process_question, question, model, answer_client, judge_client
                )
                future_to_q[future] = (idx, question)

            for future in as_completed(future_to_q):
                idx, question = future_to_q[future]
                qid = question["question_id"]
                qtype = question.get("question_type", "unknown")

                result = future.result()
                results.append(result)
                completed[qid] = True
                done_count += 1

                status = "PASS" if result.get("label") else "FAIL"
                error = f" ({result['error'][:40]})" if result.get("error") else ""
                memories = result.get("num_memories_found", "?")
                print(f"[{done_count}/{total}] {qid} ({qtype}) {status} (memories: {memories}){error}", flush=True)

        # Checkpoint after each batch
        save_checkpoint(run_id, model, completed, results)

    elapsed = time.time() - start_time
    actual_processed = len(pending)
    rate = elapsed / actual_processed if actual_processed else 0
    print(f"\nCompleted {actual_processed} questions in {elapsed:.1f}s ({rate:.2f}s/question)")

    save_checkpoint(run_id, model, completed, results)
    save_results(results, run_id, model)
    print_report(results, model=model)
    return results


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for memdio")
    parser.add_argument("--model", type=str, help="Single model to test (default: all)")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--limit", type=int, help="Limit number of questions (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for LLM calls (default: 8)")
    args = parser.parse_args()

    run_id = args.resume or str(uuid.uuid4())[:8]
    models = [args.model] if args.model else ANSWER_MODELS

    all_results = {}
    for model in models:
        results = run_benchmark(model, run_id, limit=args.limit, workers=args.workers)
        all_results[model] = results

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  CROSS-MODEL COMPARISON")
        print(f"{'=' * 60}")
        for model, results in all_results.items():
            labels = [r["label"] for r in results]
            acc = sum(labels) / len(labels) if labels else 0
            print(f"  {model:45s}: {acc:.1%}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

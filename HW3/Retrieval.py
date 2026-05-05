from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
QUESTION_COLUMNS = ["id", "prompt", "A", "B", "C", "D", "E"]
ANSWER_COLUMN = "answer"
QUERY_MODES = ("prompt_only", "prompt_with_choices")
INDEX_NAMES = ("chunks_a", "chunks_b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_RERANKERS: dict[tuple[str, str | None], CrossEncoder] = {}


def resolve_project_path(raw_path: str, label: str, must_exist: bool = True) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_match(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_tokens(text: str) -> list[str]:
    normalized = normalize_match(text)
    return normalized.split() if normalized else []


def content_keywords(text: str) -> set[str]:
    return {token for token in match_tokens(text) if len(token) > 1 and token not in STOPWORDS}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path_str: str, limit: int | None = None) -> pd.DataFrame:
    path = resolve_project_path(path_str, "Questions CSV")
    frame = pd.read_csv(path)
    missing = [column for column in QUESTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Question CSV is missing required columns: {missing}")
    if limit is not None:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def question_id(row: pd.Series, index: int) -> str:
    value = row.get("id")
    if value is None or pd.isna(value):
        return str(index)
    return str(value)


def option_text(row: pd.Series, letter: str) -> str:
    return str(row[letter]).strip()


def correct_option_text(row: pd.Series) -> str | None:
    answer = row.get(ANSWER_COLUMN)
    if answer is None or pd.isna(answer):
        return None
    answer = str(answer).strip().upper()
    if answer not in {"A", "B", "C", "D", "E"}:
        return None
    return option_text(row, answer)


def build_query(row: pd.Series, mode: str = "prompt_only") -> str:
    if mode not in QUERY_MODES:
        raise ValueError(f"Unsupported query mode: {mode}. Expected one of {QUERY_MODES}.")
    parts = [f"Question: {normalize_text(str(row['prompt']))}"]
    if mode == "prompt_with_choices":
        for letter in ("A", "B", "C", "D", "E"):
            parts.append(f"{letter}. {normalize_text(str(row[letter]))}")
    return "\n".join(parts)


def get_embeddings(
    model_name: str = "models/bge-m3",
    device: str | None = None,
    batch_size: int = 64,
) -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    model_path = resolve_project_path(model_name, "Embedding model")
    model_device = device or "cpu"
    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": model_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    return _EMBEDDINGS


def load_reranker(model_name: str = "models/ms-marco-MiniLM-L6-v2", device: str | None = None) -> CrossEncoder:
    cache_key = (model_name, device)
    if cache_key in _RERANKERS:
        return _RERANKERS[cache_key]

    model_path = resolve_project_path(model_name, "Reranker model")
    reranker = CrossEncoder(str(model_path), device=device)
    _RERANKERS[cache_key] = reranker
    return reranker


def load_vector_db(
    persist_directory: str,
    collection_name: str,
    embedding_model: str = "models/bge-m3",
    device: str | None = None,
    batch_size: int = 64,
):
    db_path = resolve_project_path(persist_directory, "Vector DB directory")
    return Chroma(
        persist_directory=str(db_path),
        collection_name=collection_name,
        embedding_function=get_embeddings(embedding_model, device=device, batch_size=batch_size),
    )


def load_default_vector_db(index_name: str = "chunks_b", device: str | None = None, batch_size: int = 64):
    return load_vector_db(f"{index_name}", index_name, device=device, batch_size=batch_size)


def retrieve_with_scores(vector_db, query: str, k: int) -> list[dict[str, Any]]:
    docs_and_scores = vector_db.similarity_search_with_score(query, k=k)
    results: list[dict[str, Any]] = []
    for rank, (document, distance) in enumerate(docs_and_scores, start=1):
        similarity = 1.0 / (1.0 + float(distance))
        results.append(
            {
                "rank": rank,
                "document": document,
                "vector_score": similarity,
                "vector_distance": float(distance),
            }
        )
    return results


def rerank_candidates(
    query: str,
    candidates: Sequence[dict[str, Any]],
    reranker: CrossEncoder,
    final_k: int,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.perf_counter()
    pairs = [[query, item["document"].page_content] for item in candidates]
    scores = reranker.predict(pairs) if pairs else []
    elapsed = time.perf_counter() - started_at

    reranked = [{**candidate, "rerank_score": float(score)} for candidate, score in zip(candidates, scores)]
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rerank_rank"] = rank
    return reranked[:final_k], elapsed


def advanced_rag_retrieve(query, db, top_k_retrieval: int = 5, top_k_rerank: int = 3):
    reranker = load_reranker()
    initial_docs = retrieve_with_scores(db, query, top_k_retrieval)
    reranked_docs, _ = rerank_candidates(query, initial_docs, reranker, top_k_rerank)
    return [item["document"] for item in reranked_docs]


def contains_correct_answer(row: pd.Series, documents: Sequence[dict[str, Any]]) -> bool | None:
    gold_text = correct_option_text(row)
    if not gold_text:
        return None
    normalized_gold = normalize_match(gold_text)
    if not normalized_gold:
        return None
    return any(normalized_gold in normalize_match(item["document"].page_content) for item in documents)


def contains_token_sequence(haystack_tokens: Sequence[str], needle_tokens: Sequence[str]) -> bool:
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    width = len(needle_tokens)
    return any(list(haystack_tokens[index : index + width]) == list(needle_tokens) for index in range(len(haystack_tokens) - width + 1))


def document_supports_correct_answer(row: pd.Series, document_payload: dict[str, Any]) -> bool | None:
    gold_text = correct_option_text(row)
    if not gold_text:
        return None
    answer_tokens = match_tokens(gold_text)
    if not answer_tokens:
        return None

    document = document_payload["document"]
    combined_text = document.page_content
    document_tokens = match_tokens(combined_text)
    if not contains_token_sequence(document_tokens, answer_tokens):
        return False

    prompt_keywords = content_keywords(str(row["prompt"]))
    support_keywords = prompt_keywords - set(answer_tokens)
    if not support_keywords:
        return True
    document_keywords = content_keywords(combined_text)
    overlap = support_keywords & document_keywords
    required_overlap = 1 if len(answer_tokens) >= 2 else min(2, len(support_keywords))
    return len(overlap) >= required_overlap


def contains_supported_answer(row: pd.Series, documents: Sequence[dict[str, Any]]) -> bool | None:
    outcome: bool | None = None
    for item in documents:
        supported = document_supports_correct_answer(row, item)
        if supported is None:
            continue
        outcome = False
        if supported:
            return True
    return outcome


def serialize_document(document: Any, score: float | None = None) -> dict[str, Any]:
    payload = {"page_content": document.page_content, "metadata": dict(document.metadata)}
    if score is not None:
        payload["score"] = float(score)
    return payload


def summarize_timings(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_seconds": 0.0, "median_seconds": 0.0}
    return {
        "mean_seconds": round(statistics.fmean(values), 4),
        "median_seconds": round(statistics.median(values), 4),
    }


def collect_rerank_cases(results: Sequence[dict[str, Any]], max_cases: int = 3) -> list[dict[str, Any]]:
    cases = []
    for result in results:
        if result.get("vector_hit_at_final_k") is False and result.get("rerank_hit_at_final_k") is True:
            cases.append(
                {
                    "question_id": result["question_id"],
                    "prompt": result["prompt"],
                    "answer": result.get("answer"),
                    "vector_top_k": result["vector_top_k"],
                    "rerank_top_k": result["rerank_top_k"],
                }
            )
        if len(cases) >= max_cases:
            break
    return cases


def build_summary(
    *,
    index_name: str,
    db_path: str,
    embedding_model: str,
    reranker_model: str,
    query_mode: str,
    question_count: int,
    initial_k: int,
    final_k: int,
    vector_recalls_at_initial_k: Sequence[bool],
    vector_hits_at_final_k: Sequence[bool],
    rerank_hits_at_final_k: Sequence[bool],
    support_recalls_at_initial_k: Sequence[bool],
    support_vector_hits_at_final_k: Sequence[bool],
    support_rerank_hits_at_final_k: Sequence[bool],
    vector_times: Sequence[float],
    rerank_times: Sequence[float],
    total_times: Sequence[float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "index_name": index_name,
        "db_path": db_path,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "query_mode": query_mode,
        "question_count": question_count,
        "initial_k": initial_k,
        "final_k": final_k,
        "latency": {
            "vector_search": summarize_timings(vector_times),
            "reranking": summarize_timings(rerank_times),
            "total": summarize_timings(total_times),
        },
    }
    if vector_recalls_at_initial_k:
        summary["vector_recall_at_initial_k"] = round(sum(vector_recalls_at_initial_k) / len(vector_recalls_at_initial_k), 4)
    if vector_hits_at_final_k:
        summary["vector_hit_at_final_k"] = round(sum(vector_hits_at_final_k) / len(vector_hits_at_final_k), 4)
    if rerank_hits_at_final_k:
        summary["rerank_hit_at_final_k"] = round(sum(rerank_hits_at_final_k) / len(rerank_hits_at_final_k), 4)
    if support_recalls_at_initial_k:
        summary["support_recall_at_initial_k"] = round(sum(support_recalls_at_initial_k) / len(support_recalls_at_initial_k), 4)
    if support_vector_hits_at_final_k:
        summary["support_vector_hit_at_final_k"] = round(sum(support_vector_hits_at_final_k) / len(support_vector_hits_at_final_k), 4)
    if support_rerank_hits_at_final_k:
        summary["support_rerank_hit_at_final_k"] = round(sum(support_rerank_hits_at_final_k) / len(support_rerank_hits_at_final_k), 4)
    return summary


def evaluate_index(args: argparse.Namespace, index_name: str) -> dict[str, Any]:
    questions = load_questions(args.questions, args.question_limit)
    vector_db = load_vector_db(
        persist_directory=f"{args.db_root}/{index_name}",
        collection_name=index_name,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    reranker = load_reranker(args.reranker_model, args.device)

    results = []
    vector_times: list[float] = []
    rerank_times: list[float] = []
    total_times: list[float] = []
    vector_recalls_at_initial_k: list[bool] = []
    vector_hits_at_final_k: list[bool] = []
    rerank_hits_at_final_k: list[bool] = []
    support_recalls_at_initial_k: list[bool] = []
    support_vector_hits_at_final_k: list[bool] = []
    support_rerank_hits_at_final_k: list[bool] = []

    for row_index, (_, row) in enumerate(questions.iterrows()):
        query = build_query(row, mode=args.query_mode)

        started_at = time.perf_counter()
        vector_started_at = time.perf_counter()
        vector_top_k = retrieve_with_scores(vector_db, query, args.initial_k)
        vector_elapsed = time.perf_counter() - vector_started_at

        vector_eval_top_k = vector_top_k[: args.final_k]
        rerank_top_k, rerank_elapsed = rerank_candidates(query, vector_top_k, reranker, args.final_k)
        total_elapsed = time.perf_counter() - started_at

        vector_recall_at_initial_k = contains_correct_answer(row, vector_top_k)
        vector_hit_at_final_k = contains_correct_answer(row, vector_eval_top_k)
        rerank_hit_at_final_k = contains_correct_answer(row, rerank_top_k)
        support_recall_at_initial_k = contains_supported_answer(row, vector_top_k)
        support_vector_hit_at_final_k = contains_supported_answer(row, vector_eval_top_k)
        support_rerank_hit_at_final_k = contains_supported_answer(row, rerank_top_k)

        for value, bucket in (
            (vector_recall_at_initial_k, vector_recalls_at_initial_k),
            (vector_hit_at_final_k, vector_hits_at_final_k),
            (rerank_hit_at_final_k, rerank_hits_at_final_k),
            (support_recall_at_initial_k, support_recalls_at_initial_k),
            (support_vector_hit_at_final_k, support_vector_hits_at_final_k),
            (support_rerank_hit_at_final_k, support_rerank_hits_at_final_k),
        ):
            if value is not None:
                bucket.append(value)

        vector_times.append(vector_elapsed)
        rerank_times.append(rerank_elapsed)
        total_times.append(total_elapsed)

        results.append(
            {
                "question_id": question_id(row, row_index),
                "prompt": row["prompt"],
                "query": query,
                "answer": row.get(ANSWER_COLUMN),
                "vector_recall_at_initial_k": vector_recall_at_initial_k,
                "vector_hit_at_final_k": vector_hit_at_final_k,
                "rerank_hit_at_final_k": rerank_hit_at_final_k,
                "support_recall_at_initial_k": support_recall_at_initial_k,
                "support_vector_hit_at_final_k": support_vector_hit_at_final_k,
                "support_rerank_hit_at_final_k": support_rerank_hit_at_final_k,
                "latency_seconds": {
                    "vector_search": round(vector_elapsed, 4),
                    "reranking": round(rerank_elapsed, 4),
                    "total": round(total_elapsed, 4),
                },
                "vector_top_k": [
                    {
                        "rank": item["rank"],
                        "vector_score": item["vector_score"],
                        "vector_distance": item["vector_distance"],
                        "document": serialize_document(item["document"]),
                    }
                    for item in vector_top_k
                ],
                "rerank_top_k": [
                    {
                        "rank": item["rerank_rank"],
                        "vector_rank": item["rank"],
                        "vector_score": item["vector_score"],
                        "vector_distance": item["vector_distance"],
                        "rerank_score": item["rerank_score"],
                        "document": serialize_document(item["document"]),
                    }
                    for item in rerank_top_k
                ],
            }
        )

    summary = build_summary(
        index_name=index_name,
        db_path=f"{args.db_root}/{index_name}",
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        query_mode=args.query_mode,
        question_count=len(results),
        initial_k=args.initial_k,
        final_k=args.final_k,
        vector_recalls_at_initial_k=vector_recalls_at_initial_k,
        vector_hits_at_final_k=vector_hits_at_final_k,
        rerank_hits_at_final_k=rerank_hits_at_final_k,
        support_recalls_at_initial_k=support_recalls_at_initial_k,
        support_vector_hits_at_final_k=support_vector_hits_at_final_k,
        support_rerank_hits_at_final_k=support_rerank_hits_at_final_k,
        vector_times=vector_times,
        rerank_times=rerank_times,
        total_times=total_times,
    )

    artifact_dir = ensure_directory(resolve_project_path(args.artifact_dir, "Artifact directory", must_exist=False) / index_name)
    results_path = artifact_dir / "results.jsonl"
    summary_path = artifact_dir / "summary.json"
    cases_path = artifact_dir / "rerank_cases.json"

    write_jsonl(results_path, results)
    write_json(summary_path, summary)
    write_json(cases_path, collect_rerank_cases(results))

    print(f"[{index_name}] vector_hit@{args.final_k}: {summary.get('vector_hit_at_final_k', 'n/a')}")
    print(f"[{index_name}] rerank_hit@{args.final_k}: {summary.get('rerank_hit_at_final_k', 'n/a')}")
    print(f"[{index_name}] summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vector-only retrieval vs vector + reranking.")
    parser.add_argument("--questions", default="data/train.csv", help="Question CSV path inside HW3.")
    parser.add_argument("--db-root", default="db", help="Root directory containing Chroma collections.")
    parser.add_argument("--embedding-model", default="models/bge-m3", help="Embedding model directory inside HW3.")
    parser.add_argument("--reranker-model", default="models/ms-marco-MiniLM-L6-v2", help="Reranker model directory inside HW3.")
    parser.add_argument("--initial-k", type=int, default=20, help="Top-K candidates from vector search.")
    parser.add_argument("--final-k", type=int, default=3, help="Shared cutoff K for hit-rate comparison.")
    parser.add_argument("--question-limit", type=int, default=200, help="Number of questions to evaluate.")
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="prompt_only", help="Retrieval query format.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--artifact-dir", default="artifacts/retrieval", help="Output directory inside HW3.")
    parser.add_argument("--index-name", choices=INDEX_NAMES, default=None, help="Optional single collection to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_k > args.initial_k:
        raise ValueError(f"--final-k ({args.final_k}) cannot exceed --initial-k ({args.initial_k}).")

    index_names = [args.index_name] if args.index_name else list(INDEX_NAMES)
    for index_name in index_names:
        evaluate_index(args, index_name)


if __name__ == "__main__":
    main()

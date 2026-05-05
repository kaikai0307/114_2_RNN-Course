# Homework 3 Report

## Overview

This report summarizes the indexing, retrieval, re-ranking, and generation experiments that were run in the current repository.

Important caveat:

- The current local pipeline indexes `data/train.csv` rather than an external Wikipedia corpus.
- Therefore, the results below are valid as local pipeline experiments, but they are not yet a fully faithful reproduction of the original homework setting that expects a separate knowledge base.

Even with that limitation, all core experiments required by the homework were executed:

1. chunking and index construction
2. vector search vs vector search + re-ranking
3. 50-question generation with Ollama
4. latency analysis

Artifacts produced during these runs are stored under:

- `artifacts/retrieval/`
- `artifacts/retrieval_512/`
- `artifacts/retrieval_short_query8/`
- `artifacts/generation_eval/`

## Experimental Setup

### Models

- Embedding model: `BAAI/bge-m3`
- Re-ranking model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Generation model: `mistral:7b`

### Chunking Methods

- Method A: fixed-size chunking with 10% overlap
- Method B: semantic/recursive chunking that preserves prompt-choice structure when possible

### Local Index Variants

Four indices were built:

| Chunk Size | Method A | Method B |
|---|---:|---:|
| 512 | 416 chunks | 200 chunks |
| 1024 | 254 chunks | 200 chunks |

These counts were read directly from:

- `db_512/chunks_a/chroma.sqlite3`
- `db_512/chunks_b/chroma.sqlite3`
- `db_1024/chunks_a/chroma.sqlite3`
- `db_1024/chunks_b/chroma.sqlite3`

## Part 1: Chunk Size Analysis

### Observation 1: Method A is sensitive to chunk size

- `db_512/chunks_a` produced 416 chunks
- `db_1024/chunks_a` produced 254 chunks

This means smaller fixed chunks fragment the data much more aggressively. On this dataset, many question rows are cut into multiple pieces under the 512-token setting.

### Observation 2: Method B is structurally stable

- `db_512/chunks_b` produced 200 chunks
- `db_1024/chunks_b` produced 200 chunks

Because the current corpus is question rows rather than long Wikipedia articles, Method B usually keeps one question row as one intact chunk. This prevents answer choices from being split across multiple fragments.

### Interpretation

For the current dataset:

- smaller fixed chunks are more likely to split a question and its answer options into multiple fragments
- larger fixed chunks reduce fragmentation but still do not explicitly preserve structure
- semantic/recursive chunking is the cleanest approach because it keeps the prompt and choices together

This directly addresses the homework question about chunk size:

- yes, smaller chunks can cut off useful context
- the problem is especially visible in Method A
- Method B avoids most of that damage on this dataset

## Part 2: Retrieval And Re-ranking

Two retrieval experiments were run:

1. Standard full-query retrieval
2. Short-query stress test

The second one was added because the standard setup is too easy for `chunks_b`: since the corpus contains the exact same question rows as the queries, semantic chunking often returns the exact matching row at rank 1, making the score trivially perfect.

## Part 2A: Standard Full-Query Retrieval

Settings:

- question count: 200
- query mode: `prompt_only`
- vector retrieval: top-5
- final comparison cutoff: top-1

Artifacts:

- `artifacts/retrieval_512/chunks_a/summary.json`
- `artifacts/retrieval_512/chunks_b/summary.json`
- `artifacts/retrieval/chunks_a/summary.json`
- `artifacts/retrieval/chunks_b/summary.json`

### Standard Retrieval Results

| Index | Vector Recall@5 | Vector Hit@1 | Rerank Hit@1 | Support Vector Hit@1 | Support Rerank Hit@1 |
|---|---:|---:|---:|---:|---:|
| `db_512/chunks_a` | 0.815 | 0.650 | 0.655 | 0.645 | 0.655 |
| `db_512/chunks_b` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `db_1024/chunks_a` | 0.970 | 0.925 | 0.930 | 0.925 | 0.930 |
| `db_1024/chunks_b` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Standard Retrieval Latency

| Index | Vector Search Mean | Re-ranking Mean | Total Mean |
|---|---:|---:|---:|
| `db_512/chunks_a` | 0.0294 s | 0.0237 s | 0.0531 s |
| `db_512/chunks_b` | 0.0287 s | 0.0340 s | 0.0628 s |
| `db_1024/chunks_a` | 0.0216 s | 0.0147 s | 0.0363 s |
| `db_1024/chunks_b` | 0.0191 s | 0.0162 s | 0.0353 s |

### Standard Retrieval Interpretation

- Re-ranking gives a small but real gain for Method A.
- For `db_512/chunks_a`, Hit@1 improves from 0.650 to 0.655.
- For `db_1024/chunks_a`, Hit@1 improves from 0.925 to 0.930.
- Method B remains perfect in this setting because the query and indexed chunk are almost identical at the row level.

This means the standard full-query experiment does show re-ranking helping, but it does not adequately stress Method B.

## Part 2B: Short-Query Stress Test

To avoid trivial exact-match retrieval, a second diagnostic experiment was run.

Settings:

- query = only the first 8 tokens of the original question
- vector retrieval: top-20
- final comparison cutoff: top-1
- question count: 200

Artifacts:

- `artifacts/retrieval_short_query8/512_a/summary.json`
- `artifacts/retrieval_short_query8/512_b/summary.json`
- `artifacts/retrieval_short_query8/1024_a/summary.json`
- `artifacts/retrieval_short_query8/1024_b/summary.json`

### Short-Query Results

| Index | Vector Hit@1 | Rerank Hit@1 | Support Vector Hit@1 | Support Rerank Hit@1 | Better Cases | Worse Cases |
|---|---:|---:|---:|---:|---:|---:|
| `db_512/chunks_a` | 0.595 | 0.635 | 0.595 | 0.635 | 10 | 2 |
| `db_512/chunks_b` | 0.910 | 0.965 | 0.910 | 0.965 | 11 | 0 |
| `db_1024/chunks_a` | 0.845 | 0.890 | 0.845 | 0.890 | 9 | 0 |
| `db_1024/chunks_b` | 0.910 | 0.965 | 0.910 | 0.965 | 11 | 0 |

### Short-Query Latency

| Index | Vector Search Mean | Re-ranking Mean | Total Mean |
|---|---:|---:|---:|
| `db_512/chunks_a` | 0.0277 s | 0.0366 s | 0.0644 s |
| `db_512/chunks_b` | 0.0252 s | 0.0763 s | 0.1015 s |
| `db_1024/chunks_a` | 0.0241 s | 0.0502 s | 0.0742 s |
| `db_1024/chunks_b` | 0.0233 s | 0.0710 s | 0.0943 s |

### Why This Stress Test Matters

This experiment reveals the effect that the standard setting hides:

- `chunks_b` is no longer always 1.000
- re-ranking clearly improves retrieval for Method B
- in both `db_512/chunks_b` and `db_1024/chunks_b`, Hit@1 improves from 0.910 to 0.965
- the improvement comes with zero degradation cases in this diagnostic

This is the clearest evidence in the current repo that re-ranking is genuinely useful.

## Re-ranking Impact: Two Concrete Examples

The homework asks for 2 examples where vector search retrieves the wrong document at the top, but re-ranking fixes it.

Examples were taken from:

- `artifacts/retrieval_short_query8/1024_b/rerank_cases.json`

### Example 1

- Question ID: `1`
- Query used: `Question: Which of the following is an accurate definition`
- Gold answer: `A`
- Vector top-1 source: `row_166`
- Reranked top-1 source: `row_1`

What happened:

- Stage 1 retrieved an unrelated physics question about explicit symmetry breaking.
- Stage 2 promoted the correct dynamic-scaling question to rank 1.

### Example 2

- Question ID: `4`
- Query used: `Question: Which of the following statements accurately describes the`
- Gold answer: `D`
- Vector top-1 source: `row_10`
- Reranked top-1 source: `row_4`

What happened:

- Stage 1 retrieved an unrelated question about Fresnel and total internal reflections.
- Stage 2 promoted the correct diffraction-pattern question to rank 1.

These two cases directly satisfy the re-ranking example requirement from the PDF.

## Is Re-ranking Worth The Cost?

### Standard Setting

In the standard full-query setting:

- re-ranking gives only a small gain for Method A
- re-ranking gives no measurable gain for Method B because the task is too easy

### Stress-Test Setting

In the short-query setting:

- re-ranking gives a meaningful improvement for all four indices
- the clearest result is Method B:
  - `0.910 -> 0.965` on both 512 and 1024
  - `11` improved cases
  - `0` degraded cases

### Conclusion

Yes, re-ranking is worth the extra latency when retrieval is not a trivial exact-match task.

The extra cost is roughly:

- around `0.015 ~ 0.034 s` in the standard setup
- around `0.037 ~ 0.076 s` in the short-query stress test

That overhead is acceptable when it yields a `5.5%` absolute Hit@1 improvement for Method B under harder query conditions.

## Part 3: Generation With Ollama

A 50-question generation experiment was run with:

- model: `mistral:7b`
- retrieval index: `db_1024/chunks_b`
- vector retrieval: top-20
- reranked context: top-3

Artifacts:

- `artifacts/generation_eval/results_50_chunks_b_1024_mistral7b.jsonl`
- `artifacts/generation_eval/summary_50_chunks_b_1024_mistral7b.json`

### Generation Result

- question count: `50`
- correct: `30`
- accuracy: `0.600`

### Average Latency

| Stage | Mean Latency |
|---|---:|
| Vector Search | 0.0369 s |
| Re-ranking | 0.0749 s |
| LLM Generation | 0.6503 s |
| Total | 0.7621 s |

### Generation Interpretation

The generation accuracy is only `60.0%`, which is much lower than the retrieval hit rate. This is expected in the current setup because:

- the retrieved context is mostly the original question row itself
- the context does not explicitly contain the gold answer label
- the model still has to solve the science question instead of reading the answer from an external knowledge document

So the generation experiment is technically complete, but its accuracy is limited by the current corpus design.

## Final Conclusions

### Chunking

- Method B is the better chunking strategy for this dataset because it preserves prompt-choice structure.
- Method A is more sensitive to chunk size and can fragment the context.
- 1024 works better than 512 for Method A because it reduces fragmentation.

### Retrieval

- Re-ranking consistently helps Method A in the standard full-query setting.
- Under the short-query stress test, re-ranking strongly improves Method B as well.
- The best non-trivial retrieval result in this repo is:
  - `db_1024/chunks_b`
  - short-query stress test
  - `Vector Hit@1 = 0.910`
  - `Rerank Hit@1 = 0.965`

### Generation

- On 50 questions with `mistral:7b`, the current pipeline achieved `60.0%` accuracy.
- Generation is the slowest stage by far, averaging `0.6503 s` per question.

## Remaining Limitation

The only major remaining limitation is the corpus itself:

- the assignment expects an external Wikipedia knowledge base
- the current local experiment indexes `train.csv`

So the engineering conclusions about chunking and re-ranking are useful, but the final assignment-grade RAG evaluation would be stronger if the same pipeline were rerun on the intended Wikipedia subset.

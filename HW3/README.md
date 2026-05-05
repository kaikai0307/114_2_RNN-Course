# HW3

RAG for science multiple-choice question answering.

## Main Files

- `Chunking.py`: build Method A / Method B chunks
- `DB.py`: build Chroma vector databases
- `Retrieval.py`: evaluate vector search and re-ranking
- `Generation.py`: run retrieval + Ollama generation
- `run_generation_with_ollama.sh`: start Ollama if needed and run generation
- `report.md`: experiment summary

## Environment

Recommended Python:

```bash
/home/jiakai/miniconda3/envs/rnn/bin/python
```

## Build Vector DB

```bash
cd /ssd6/jiakai/114_2_RNN/HW3
/home/jiakai/miniconda3/envs/rnn/bin/python DB.py
```

This builds:

- `db_512/chunks_a`
- `db_512/chunks_b`
- `db_1024/chunks_a`
- `db_1024/chunks_b`

## Run Retrieval

1024 setting:

```bash
/home/jiakai/miniconda3/envs/rnn/bin/python Retrieval.py --db-root db_1024 --question-limit 200 --query-mode prompt_only --initial-k 5 --final-k 1 --device cuda
```

512 setting:

```bash
/home/jiakai/miniconda3/envs/rnn/bin/python Retrieval.py --db-root db_512 --artifact-dir artifacts/retrieval_512 --question-limit 200 --query-mode prompt_only --initial-k 5 --final-k 1 --device cuda
```

## Run Generation

```bash
cd /ssd6/jiakai/114_2_RNN/HW3
OLLAMA_MODEL=mistral:7b ./run_generation_with_ollama.sh
```

## Reports

- Markdown report: `report.md`
- LaTeX source: `Homework3_report.tex`
- Compiled PDF: `Homework3_report.pdf`

## Notes

- Current local experiment indexes `data/train.csv`, not Wikipedia.
- `mistral:7b` is the confirmed local Ollama model name.

#!/usr/bin/env bash

set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b}"
export OLLAMA_MODEL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
DEFAULT_OLLAMA_BIN="${PROJECT_DIR}/ollama/bin/ollama"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -x "${DEFAULT_OLLAMA_BIN}" ]]; then
  OLLAMA_BIN="${OLLAMA_BIN:-${DEFAULT_OLLAMA_BIN}}"
else
  OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
fi

if ! command -v "${OLLAMA_BIN}" >/dev/null 2>&1 && [[ ! -x "${OLLAMA_BIN}" ]]; then
  echo "Ollama binary not found. Set OLLAMA_BIN or ensure ollama is installed." >&2
  exit 1
fi

OLLAMA_STARTED_BY_SCRIPT=0
OLLAMA_PID=""

cleanup() {
  if [[ "${OLLAMA_STARTED_BY_SCRIPT}" == "1" && -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

ollama_ready() {
  curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

start_ollama_if_needed() {
  if ollama_ready; then
    echo "=============================="
    echo "Ollama is already running at ${OLLAMA_URL}."
    echo "=============================="
    return
  fi

  echo "====================================="
  echo "Starting Ollama from ${OLLAMA_BIN}..."
  echo "====================================="
  (
    cd "${PROJECT_DIR}"
    exec "${OLLAMA_BIN}" serve
  ) >"${PROJECT_DIR}/ollama_serve.log" 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED_BY_SCRIPT=1

  for _ in $(seq 1 60); do
    if ollama_ready; then
      echo "Ollama is ready."
      return
    fi
    sleep 1
  done

  echo "Ollama did not become ready within 60 seconds. Check ${PROJECT_DIR}/ollama_serve.log." >&2
  exit 1
}

main() {
  cd "${PROJECT_DIR}"
  start_ollama_if_needed

  if [[ "$#" -gt 0 ]]; then
    echo "Generation.py currently uses in-file settings and does not accept CLI arguments."
    echo "Ignoring arguments: $*"
  fi

  echo "======================="
  echo "Running Generation.py"
  "${PYTHON_BIN}" Generation.py
}

main "$@"

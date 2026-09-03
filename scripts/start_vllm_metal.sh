#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_METAL_VENV="${VLLM_METAL_VENV:-$PROJECT_ROOT/../vllm-metal/.venv-vllm-metal}"
VLLM_METAL_BIN="$VLLM_METAL_VENV/bin/vllm"
EMBEDDING_GGUF_PATH="${OAMB_VLLM_METAL_GGUF_PATH:-$HOME/.cache/qmd/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
EMBEDDING_HF_DIR="${OAMB_VLLM_METAL_HF_DIR:-$HOME/.cache/oamb-vllm-metal/Qwen3-Embedding-0.6B-hf}"

if [[ ! -x "$VLLM_METAL_BIN" ]]; then
  echo "vLLM-Metal executable not found: $VLLM_METAL_BIN" >&2
  echo "Set VLLM_METAL_VENV to the installed vLLM-Metal environment." >&2
  exit 1
fi

if [[ ! -f "$EMBEDDING_GGUF_PATH" ]]; then
  echo "cached Qwen3 embedding GGUF not found: $EMBEDDING_GGUF_PATH" >&2
  echo "Set OAMB_VLLM_METAL_GGUF_PATH to the existing GGUF file." >&2
  exit 1
fi

if [[ ! -f "$EMBEDDING_HF_DIR/config.json" || ! -f "$EMBEDDING_HF_DIR/tokenizer_config.json" ]]; then
  echo "cached Qwen3 embedding tokenizer/config not found: $EMBEDDING_HF_DIR" >&2
  echo "Set OAMB_VLLM_METAL_HF_DIR to the existing Hugging Face metadata directory." >&2
  exit 1
fi

exec env \
  VLLM_METAL_BUILD_FROM_SOURCE=1 \
  VLLM_METAL_MEMORY_FRACTION=0.06 \
  "$VLLM_METAL_BIN" serve "$EMBEDDING_GGUF_PATH" \
  --tokenizer "$EMBEDDING_HF_DIR" \
  --hf-config-path "$EMBEDDING_HF_DIR" \
  --hf-overrides '{"matryoshka_dimensions":[1024]}' \
  --runner pooling \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --additional-config '{"turboquant":true,"k_quant":"q8_0","v_quant":"q8_0"}' \
  --host 127.0.0.1 \
  --port 18000 \
  --served-model-name qwen3-embedding:0.6b

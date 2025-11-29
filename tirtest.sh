#!/bin/bash
set -e

MODEL_PATH=/root/autodl-tmp/models/Qwen_Qwen3-4B
PORT=8000

mkdir -p logs

echo "[Start] vLLM server with model: ${MODEL_PATH}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --tensor-parallel-size 4 \
  --max-model-len 2304 \
  --port ${PORT} \
  --trust-remote-code \
  --served-model-name Qwen_Qwen3-4B \
  --gpu-memory-utilization 0.8 \
  > logs/qwen3_server.log 2>&1 &

SERVER_PID=$!
echo "[Info] vLLM server pid=${SERVER_PID}"
sleep 20   

# SQuAD v2
python src/eval_qwen3_agent.py \
  --dataset squadv2 \
  --source json \
  --input /root/autodl-tmp/TIRBench/data/squadv2.json \
  --decision squad_v2 \
  --squad_f1_threshold 0.7

# Hotpot
python src/eval_qwen3_agent.py \
  --dataset hotpot \
  --source json \
  --input /root/autodl-tmp/TIRBench/data/hotpot.json \
  --decision hotpot \
  --hotpot_f1_threshold 0.7   
# NQ
python src/eval_qwen3_agent.py \
  --dataset nq \
  --source json \
  --input /root/autodl-tmp/TIRBench/data/nq.json \
  --decision nq \
  --nq_f1_threshold 0.7

# # AQuA
# python src/eval_qwen3_agent.py \
#   --dataset aqua \
#   --source json \
#   --input /root/autodl-tmp/TIRBench/data/aqua.json

# # GSM8K
# python src/eval_qwen3_agent.py \
#   --dataset gsm8k \
#   --source json \
#   --input /root/autodl-tmp/TIRBench/data/gsm8k.json

# # MATH
# python src/eval_qwen3_agent.py \
#   --dataset math \
#   --source json \
#   --input /root/autodl-tmp/TIRBench/src/math.json

########################################
# 跑完关掉 server
########################################
echo "[Stop] killing vLLM server pid=${SERVER_PID}"
kill ${SERVER_PID} || true

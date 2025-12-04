# SQuAD v2 (dev)
python /root/autodl-tmp/TIRBench/src/data_cleaning.py \
  --backend vllm \
  --dataset squadv2 \
  --source json \
  --input /root/autodl-tmp/TIRBench/src/datasets/SQuAD/dev-v2.0.json \
  --model /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --tokenizer_path /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --batch_size 64 \
  --max_new_tokens 8192 \
  --temperature 0.0 --top_p 1.0 \
  --decision squad_v2 \
  --squad_f1_threshold 0.7 \
  --include_unanswerable_hint \
  --store_think --store_prompt --store_context 256 \
  --run_tag squadv2_bs12_mnt64_t0 \
  --output /root/autodl-tmp/TIRBench/data/squadv2.json \
  --tp 4 \
  --no_logs \
  # --max_eval_examples 200 \

# HotpotQA (distractor dev)
python /root/autodl-tmp/TIRBench/src/data_cleaning.py \
  --backend vllm \
  --dataset hotpot \
  --source json \
  --input /root/autodl-tmp/TIRBench/src/datasets/HotpotQA/hotpot_dev_distractor_v1.json \
  --model /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --tokenizer_path /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --batch_size 64 \
  --max_new_tokens 8192 \
  --temperature 0.0 --top_p 1.0 \
  --decision hotpot \
  --hotpot_f1_threshold 0.7 \
  --include_unanswerable_hint \
  --store_think --store_prompt --store_context 256 \
  --run_tag hotpot_bs8_mnt128_t0 \
  --output /root/autodl-tmp/TIRBench/data/hotpot.json \
  --tp 4 \
  --no_logs \
  # --max_eval_examples 200 \
# Natural Questions (dev jsonl)
python /root/autodl-tmp/TIRBench/src/data_cleaning.py \
  --backend vllm \
  --dataset nq \
  --source json \
  --input "/root/autodl-tmp/TIRBench/src/datasets/Natural_Question/nq-dev-all.jsonl" \
  --model /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --tokenizer_path /root/autodl-tmp/models/Qwen_Qwen3-4B \
  --batch_size 64 \
  --max_new_tokens 8192 \
  --temperature 0.0 --top_p 1.0 \
  --decision nq \
  --nq_f1_threshold 0.7 \
  --include_unanswerable_hint \
  --store_think --store_prompt --store_context 256 \
  --run_tag nq_bs6_mnt64_t0 \
  --output /root/autodl-tmp/TIRBench/data/nq.json \
  --tp 4 \
  --no_logs \

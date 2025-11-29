MODEL_PATH=/root/autodl-tmp/models/Qwen_Qwen3-4B/
TOKENIZER_PATH=$MODEL_PATH

# AQuA
python ./src/data_cleaning.py \
  --backend vllm \
  --dataset omini \
  --source json \
  --input /root/autodl-tmp/TIRBench/data/omini_500.json \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --tp 4 \
  --output /root/autodl-tmp/data \
  --eval_only \
  --max_new_tokens 2048

# # GSM8K
# python ./src/data_cleaning.py \
#   --backend vllm \
#   --dataset gsm8k \
#   --source json \
#   --input /root/autodl-tmp/TIRBench/data/gsm8k_500.json \
#   --model $MODEL_PATH \
#   --tokenizer_path $TOKENIZER_PATH \
#   --batch_size 8 \
#   --tp 4 \
#   --output /root/autodl-tmp/data \
#   --eval_only \
#   --max_new_tokens 1024
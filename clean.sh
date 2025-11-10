MODEL_PATH=/root/autodl-tmp/models/Qwen_Qwen3-4B/
TOKENIZER_PATH=$MODEL_PATH

# AQuA
python data_cleaning.py \
  --backend vllm \
  --dataset aqua \
  --source json \
  --input datasets/AQuA/validation-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --output /root/autodl-tmp/data/aqua.json \
  --tp 4

# GSM8K
python data_cleaning.py \
  --backend vllm \
  --dataset gsm8k \
  --source json \
  --input datasets/GSM8k/main/test-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --output /root/autodl-tmp/data/gsm8k.json \
  --tp 4

# MATH
python data_cleaning.py \
  --backend vllm \
  --dataset math \
  --source json \
  --input datasets/MATH/test-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --output /root/autodl-tmp/data/math.json \
  --tp 4
MODEL_PATH=./Qwen3-VL-4B-Thinking
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
  --output outputs/aqua.json \
  --tp 8

# GSM8K
python data_cleaning.py \
  --backend vllm \
  --dataset gsm8k \
  --source json \
  --input datasets/GSM8k/main/test-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --output outputs/gsm8k.json \
  --tp 8

# MATH
python data_cleaning.py \
  --backend vllm \
  --dataset math \
  --source json \
  --input datasets/MATH/test-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 4 \
  --output outputs/math.json \
  --tp 8
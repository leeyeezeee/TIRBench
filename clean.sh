MODEL_PATH=./Qwen3-VL-4B-Thinking
TOKENIZER_PATH=$MODEL_PATH

# 1. AQuA
python data_cleaning.py \
  --backend vllm \
  --dataset aqua \
  --source hf \
  --split validation \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --max_input_tokens 2048 \
  --max_new_tokens 64 \
  --output outputs/aqua.json 
  --tp 8
# 2. GSM8K
python data_cleaning.py \
  --backend vllm \
  --dataset gsm8k \
  --source hf \
  --split test \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --max_input_tokens 2048 \
  --max_new_tokens 64 \
  --output outputs/gsm8k.json 
  --tp 8
# 3. MATH
python data_cleaning.py \
  --backend vllm \
  --dataset math \
  --source hf \
  --split validation \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 4 \
  --max_input_tokens 2048 \
  --max_new_tokens 64 \
  --output outputs/math.json 
  --tp 8
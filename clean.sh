MODEL_PATH=/root/autodl-tmp/models/Qwen_Qwen3-4B/
TOKENIZER_PATH=$MODEL_PATH

python ./src/data_cleaning.py \
  --backend vllm \
  --dataset omini \
  --source json \
  --input /root/autodl-tmp/TIRBench/src/datasets/OminiMATH/test-00000-of-00001.parquet \
  --model $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size 8 \
  --output /root/autodl-tmp/TIRBench/data/omini.json \
  --tp 4 \
  --max_new_tokens 2048

# # wmdp mmlu baseline

# model_name=meta-llama/Llama-3.1-8B-Instruct
# # GPUs_per_model=1
# # model_replicas=1
# # lm_eval --model vllm \
# #     --model_args pretrained={$model_name},tensor_parallel_size={$GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={$model_replicas} \
# #     --tasks mmlu \
# #     --batch_size auto

# export OPENAI_API_KEY=token-abc123
# api_base=http://localhost:8000/v1
# lm_eval --model local-completions --tasks mmlu --model_args model={$model_name},base_url={$api_base}/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16

# serve baseline models
OUTLINES_CACHE_DIR="/tmp/outlines_cache" CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --api-key token-abc123
OUTLINES_CACHE_DIR="/tmp/outlines_cache" CUDA_VISIBLE_DEVICES=1 vllm serve GraySwanAI/Llama-3-8B-Instruct-RR --host 0.0.0.0 --port 8001 --tensor-parallel-size 1 --api-key token-abc123
OUTLINES_CACHE_DIR="/tmp/outlines_cache" vllm serve Qwen/Qwen2.5-72B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --api-key token-abc123

# get scores for jailbreaking
# get response for strong_reject
python run_baseline_llamaguard_v2.py

# get scores for strong_reject
# cd agentic-safety/evals/strong_reject/
cd evals/strong_reject/
python eval_strongreject.py

# get scores for false_refusal
cd agentic-safety/evals/false_refusal/
python eval_false_refusal.py

# need to config the .env file for different apis

# serve a model using vllm
# CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --api-key token-abc123
# curl -X GET http://localhost:8000/v1/models -H "Authorization: Bearer token-abc123" 

# run hosted vllm
model_name=hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
# model_name=hosted_vllm/Qwen/Qwen2.5-72B-Instruct
api_base=http://localhost:8000/v1
# api_base=http://localhost:8001/v1

# run deepseek
# model_name=deepseek/deepseek-chat
# api_base=https://api.deepseek.com/v1

# run openai
# model_name=openai/gpt-4o-2024-11-20
# api_base=https://api.openai.com/v1


# wmdp benchmark
# split="wmdp_chem"
# python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base
# run without optimization
# python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base enable_dspy_optimization=false

# splits=("wmdp_bio" "wmdp_cyber" "mmlu")
# splits=("wmdp_bio" "wmdp_chem" "wmdp_cyber" "mmlu")
# # Loop through each split
# for split in "${splits[@]}"; do
#   # Generate the output file name
# #   output_file="$(date +%Y%m%d-%H%M%S)_run_dspy_${split}_$(basename "$model_name").txt"
#   output_file="$(date +%Y%m%d-%H%M%S)_run_dspy_no_optimization_${split}_$(basename "$model_name").txt"
  
#   # Execute the command
# #   yes | nohup python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base > $output_file 2>&1 &
#   yes | nohup python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base enable_dspy_optimization=false > $output_file 2>&1 &
#   echo "Started task for split: $split, output logging to: $output_file"
# done

# run mt-bench
# split=mt_bench
# judge_model_name=meta-llama/Llama-3.1-8B-Instruct # do not include provider as suffix
# python run_mt_bench.py run=$split unlearning=default \
#     model.model_name=$model_name model.judge_model_name=$judge_model_name \
#     model.api_base=$api_base model.judge_api_base=$api_base \
#     run.judge_only=true run.judge_reference_mappings.gpt-4=$judge_model_name

# output_file="$(date +%Y%m%d-%H%M%S)_run_dspy_${split}_$(basename "$model_name").txt"
# yes | nohup python run_dspy_mt_bench.py run=$split unlearning=default \
#     model.model_name=$model_name model.judge_model_name=$judge_model_name \
#     model.api_base=$api_base model.judge_api_base=$api_base \
#     run.judge_only=true run.judge_reference_mappings.gpt-4=$judge_model_name > $output_file 2>&1 &



# run harry potter
# python run_harry_potter.py run=harry_potter model.model_name=$model_name model.api_base=$api_base


# run tofu 
# python run_tofu_server.py --hf_cache_dir=$HF_HOME/hub
# python run_tofu.py run=tofu unlearning.unlearning_text_file=unlearning_tofu_01.txt run.data.subset=forget01



# run jailbreak
# split=strong_reject
split=false_refusal
# python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base
yes | nohup python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base > $(date +%Y%m%d-%H%M%S)_run_jailbreak_${split}_$(basename "$model_name").txt 2>&1 &
# python run_jailbreak.py run=$split model.model_name=$model_name model.api_base=$api_base enable_dspy_optimization=false


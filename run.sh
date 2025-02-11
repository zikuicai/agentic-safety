# need to config the .env file for different apis

# serve a model using vllm
# OUTLINES_CACHE_DIR="/tmp/outlines_cache" vllm serve Qwen/Qwen2.5-72B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --api-key token-abc123
# OUTLINES_CACHE_DIR="/tmp/outlines_cache" CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --api-key token-abc123
# CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --api-key token-abc123
# curl -X GET http://localhost:8000/v1/models -H "Authorization: Bearer token-abc123" 
# ssh XXXX -L 8000:0.0.0.0:8000
# ssh XXXX $(for i in {8000..8003}; do echo -n " -L $i:0.0.0.0:$i"; done)

# run hosted vllm
model_name=hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
api_base=http://localhost:8000/v1

# model_name=hosted_vllm/Qwen/Qwen2.5-72B-Instruct
# api_base=http://localhost:8001/v1
exp_version=0

# run deepseek
# model_name=deepseek/deepseek-chat
# model_name=deepseek/deepseek-reasoner
# api_base=https://api.deepseek.com/v1
# exp_version=0


# model_name=openai/gpt-4o-2024-11-20
# model_name=openai/gpt-4o-mini
# api_base=https://api.openai.com/v1
# exp_version=0


# wmdp benchmark
split="wmdp_chem"
python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base
# run without optimization
# python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base enable_dspy_optimization=false
# yes | python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base model.use_cache=false model.exp_version=$exp_version
# yes | python run_wmdp_baseline.py run=$split model.model_name=$model_name model.api_base=$api_base model.use_cache=false model.exp_version=$exp_version
# # model.use_cache=false

# splits=("wmdp_bio" "wmdp_cyber" "mmlu")
# # splits=("wmdp_bio" "wmdp_chem" "wmdp_cyber" "mmlu")
# # Loop through each split
# for split in "${splits[@]}"; do
# #   Generate the output file name
#   output_file="$(date +%Y%m%d-%H%M%S)_run_dspy_${split}_$(basename "$model_name").txt"
# #   output_file="$(date +%Y%m%d-%H%M%S)_run_dspy_no_optimization_${split}_$(basename "$model_name").txt"
  
#   # Execute the command
# #   yes | nohup python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base model.exp_version=$exp_version > $output_file 2>&1 &
#   yes | nohup python run_wmdp.py run=$split model.model_name=$model_name model.api_base=$api_base model.exp_version=$exp_version enable_dspy_optimization=false > $output_file 2>&1 &
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
# exp_version=0
# split=false_refusal
# # exp_version=1

# python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base model.use_cache=false model.exp_version=$exp_version
# python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base model.exp_version=$exp_version

# yes | nohup python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base > $(date +%Y%m%d-%H%M%S)_run_jailbreak_${split}_$(basename "$model_name").txt 2>&1 &
# python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base
# model.use_cache=false


# run jailbreak rapid response
# split=rapid_response
# exp_version=2
# need to change config yaml file for different n, and change num_examples in the run py file
# # python run_jailbreak_rapid.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base model.use_cache=false model.exp_version=$exp_version
# output_file="logs/$(date +%Y%m%d-%H%M%S)_run_jailbreak_${split}_$(basename "$model_name").txt"
# yes | nohup python run_jailbreak_rapid.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base model.exp_version=$exp_version > $output_file 2>&1 &

# splits=("strong_reject" "false_refusal")
# exp_version=0
# # Loop through each split
# for split in "${splits[@]}"; do
#   # Generate the output file name
#   output_file="logs/$(date +%Y%m%d-%H%M%S)_run_dspy_${split}_$(basename "$model_name").txt"
  
#   # Execute the command
#   yes | nohup python run_jailbreak.py run=$split unlearning=$split model.model_name=$model_name model.api_base=$api_base model.exp_version=$exp_version > $output_file 2>&1 &
  
#   echo "Started task for split: $split, output logging to: $output_file"
# done

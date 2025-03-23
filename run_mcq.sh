#!/bin/bash

BASE_PORT=11110
TMP_CONFIG_DIR="configs.tmp"

RUNS=("wmdp_chem" "wmdp_bio" "wmdp_cyber" "mmlu")

MODES=("base" "prompting" "filtering" "dspy-base" "dspy-json")
CUDA_DEVICES="0,1,2,3"

PIDS=()

MODEL=$1
if [ -z "$MODEL" ]; then
  echo "Model is empty. Please provide the model name."
  exit 1
fi

wait_for_llm() {
  while ! nc -z localhost $BASE_PORT; do
    echo "Waiting for vLLM to come online..."
    sleep 5
  done
  echo "vLLM is online."
}
wait_for_llm

for run in "${RUNS[@]}"; do
  for mode in "${MODES[@]}"; do
    port=$BASE_PORT

    model_name_for_logs=$(basename $MODEL)
    echo "Using model name: $MODEL"

    datetime_str=$(date '+%Y-%m-%d_%H-%M-%S' | tr ' ' '-' | tr ':' '-')
    log_file="$(date +%s)-${run}-${mode}-p${port}-${datetime_str}.log"
    logs_dir=$(pwd)/logs1/${run}_${mode}/${model_name_for_logs}
    mkdir -p $logs_dir
    echo "Logging to $logs_dir/$log_file"

    api_base="http://localhost:$BASE_PORT/v1"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python run_unlearning_mcq.py +data=$run +defense=unl_wmdp mode=$mode model.model_name=$MODEL model.api_base=$api_base > $logs_dir/$log_file 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "Started process for ${run} with mode ${mode} and PID $pid"
  done
done

echo "To kill all processes, run: kill ${PIDS[@]}"
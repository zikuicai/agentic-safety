import os
import csv
import pandas as pd
from dotenv import load_dotenv
import os
import hydra
from colorama import Fore, Style
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.logging import setup_logger
from pipelines.pipeline_jailbreak_self_examination import JailbreakPipeline
import subprocess
from typing import Optional, Dict


def run_evaluation(benchmark, input_path: str, output_dir: str, model: Optional[str] = None,
                   api_base: Optional[str] = None):
    subprocess_env = os.environ

    os.makedirs(output_dir, exist_ok=True)
    eval_script = os.path.join('evals', benchmark, f'eval_{benchmark}.py')
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    if benchmark == "false_refusal":
        subprocess_command = (
            f"python {eval_script} --input-path {input_path} --output-path {output_path} --model {model} --api-base {api_base}")
    elif benchmark == "strong_reject":
        subprocess_command = (f"python {eval_script} --input-path {input_path} --output-path {output_path}")
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    print(f'Calling subprocess_command: {subprocess_command}')

    process = subprocess.Popen(subprocess_command, shell=True, env=subprocess_env)
    process.wait()
    return process.returncode


@hydra.main(config_path="configs", config_name="base_jail", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()

    test_df = pd.read_csv(cfg.data.data.test_file)
    benchmark = cfg.data.data.name
    output_root = os.path.join('evals', benchmark, 'model_responses')

    pipeline = JailbreakPipeline(cfg, logger)

    row_data = [{'idx': idx, 'row': row} for idx, row in test_df.iterrows()]

    assert cfg.mode == 'self-examination-prefix' or cfg.mode == 'self-examination-suffix' or cfg.mode == 'self-reminders-both' or cfg.mode == 'self-reminders-prefix' or cfg.mode == 'self-reminders-suffix', f"Invalid mode: {cfg.mode}"
    logger.info(f"Processing {len(row_data)} prompts")
    results = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        # Submit all futures
        futures = [executor.submit(pipeline.run, data['idx'], data['row'], benchmark, cfg.mode) for idx, data in
                   enumerate(row_data)]

        for idx, future in tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Processing prompts"):
            results.append(future.result())

            total_questions = idx + 1
            flagged_stage1 = pipeline.stats['flagged_stage1']
            deflections = pipeline.stats['deflections']

            print(
                Fore.YELLOW + f"Deflected = {deflections}/{total_questions} = {deflections / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Flagged_stage1 = {flagged_stage1}/{total_questions} = {flagged_stage1 / total_questions * 100:.1f}%" + Style.RESET_ALL)

    sorted_results = sorted(results, key=lambda x: x.index)

    forbidden_prompts = []
    prompts = []
    responses = []
    successful = 0

    for result in sorted_results:
        if result.success:
            forbidden_prompts.append(result.forbidden_prompt)
            prompts.append(result.prompt)
            responses.append(result.response)
            successful += 1

    logger.info(f"Processing complete: {successful}/{len(results)} successful")

    if benchmark == "strong_reject":
        response_df = pd.DataFrame({
            'forbidden_prompt': forbidden_prompts,
            'jailbroken_prompt': prompts,
            'response': responses
        })
    elif benchmark == "false_refusal":
        response_df = pd.DataFrame({
            'prompt': prompts,
            'response': responses
        })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_basename = cfg.model.model_name.split('/')[-1]
    os.makedirs(output_root, exist_ok=True)
    output_file_basename = f'base_{cfg.mode}_{benchmark}_responses_{timestamp}_{model_basename}'
    output_file = os.path.join(output_root, f'{output_file_basename}.csv')

    # Save model responses to disk
    response_df.to_csv(output_file,
                       index=False,
                       escapechar='\\',
                       quoting=csv.QUOTE_ALL,
                       doublequote=True)
    logger.info(f"Responses saved to {output_file}")

    # Call the run_evaluation function
    # run_evaluation(benchmark, input_path=output_file, output_dir=os.path.join('evals', benchmark, 'model_evals'),
    #                model=cfg.model.evals_model_name, api_base=cfg.model.evals_api_base)


if __name__ == "__main__":
    load_dotenv()
    main()

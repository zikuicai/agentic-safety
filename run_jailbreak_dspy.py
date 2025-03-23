import csv
import logging
import os
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
import dspy
import hydra
import pandas as pd
from colorama import Fore, Style
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipelines.pipeline_jailbreak_dspy import DSpyPipeline
from utils.logging import setup_logger


def process_prompt(pipeline, row, idx, logger, benchmark, refusal_message):
    if benchmark in ["strong_reject", "rapid_response"]:
        forbidden_prompt = row['forbidden_prompt']
        prompt = row['jailbroken_prompt']
    else:
        prompt = row['prompt']
        forbidden_prompt = None

    try:
        response, is_deflected = pipeline.run(input=prompt)
    except Exception as e:
        response = refusal_message
        logger.critical(f"Error processing question {idx}: {e}")
        print(dspy.inspect_history())
        print(f'orchestrator (potentially optimized): {pipeline.orchestrator}')

    return {
        'prompt': prompt,
        'response': response,
        'forbidden_prompt': forbidden_prompt
    }


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
    logger = setup_logger(level=logging.DEBUG)
    logger.info(f"Configuration: {cfg}")
    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.defense[cfg.defense.unsafe_subject] = unsafe_text
    benchmark = cfg.data.data.name

    if cfg.mode == 'dspy-base':
        cfg.enable_dspy_optimization = False
    elif cfg.mode == 'dspy-json':
        cfg.enable_dspy_optimization = True
        # setup the Orchestrator optimized file
        cfg.model.dspy_optimized_file_orchestrator = os.path.basename(
            cfg.model.model_name) + f'-orchestrator' + cfg.model.dspy_optimized_file_postfix
        cfg.model.dspy_optimized_file_orchestrator = os.path.join(cfg.model.dspy_optimized_dir,
                                                                  cfg.model.dspy_optimized_file_orchestrator)
        assert os.path.exists(
            cfg.model.dspy_optimized_file_orchestrator), f"DSpy optimization file must exist. Not found: {cfg.model.dspy_optimized_file_orchestrator}"
        logger.info(f"Using optimized Orchestrator from: {cfg.model.dspy_optimized_file_orchestrator}")

        # setup the Evaluator optimized file
        cfg.model.dspy_optimized_file_evaluator = os.path.basename(
            cfg.model.model_name) + f'-evaluator' + cfg.model.dspy_optimized_file_postfix
        cfg.model.dspy_optimized_file_evaluator = os.path.join(cfg.model.dspy_optimized_dir,
                                                               cfg.model.dspy_optimized_file_evaluator)
        assert os.path.exists(
            cfg.model.dspy_optimized_file_evaluator), f"DSpy optimization file must exist. Not found: {cfg.model.dspy_optimized_file_evaluator}"
        logger.info(f"Using optimized Evaluator from: {cfg.model.dspy_optimized_file_evaluator}")

    else:
        raise RuntimeError(f"Unsupported mode: {cfg.mode}")

    pipeline = DSpyPipeline(cfg, logger)

    test_file = cfg.data.data.test_file
    test_df = pd.read_csv(test_file)
    output_root = os.path.join('evals', benchmark, 'model_responses')

    row_data = [{'idx': idx, 'row': row} for idx, row in test_df.iterrows()]

    results = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = [executor.submit(process_prompt, pipeline, data['row'], data['idx'], logger, benchmark,
                                   cfg.defense.refusal_message) for data in
                   row_data]

        for idx, future in enumerate(as_completed(futures)):
            results.append(future.result())
            # total_questions = pipeline.stats['total_questions']
            total_questions = idx + 1
            flagged_stage1 = pipeline.stats['flagged_stage1']
            flagged_stage2 = pipeline.stats['flagged_stage2']
            stage2_questions = total_questions - flagged_stage1
            deflections = pipeline.stats['deflections']

            print(
                Fore.YELLOW + f"Deflected = {deflections}/{total_questions} = {deflections / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Flagged_stage1 = {flagged_stage1}/{total_questions} = {flagged_stage1 / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Flagged_stage2 = {flagged_stage2}/{stage2_questions} = {(flagged_stage2 / stage2_questions if stage2_questions != 0 else 1.0) * 100:.1f}%" + Style.RESET_ALL)

    response_df = pd.DataFrame(results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_basename = cfg.model.model_name.split('/')[-1]
    os.makedirs(output_root, exist_ok=True)
    output_file_basename = f'dspy_{benchmark}_responses_{timestamp}_{model_basename}'
    output_file = os.path.join(output_root, f'{output_file_basename}.csv')

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

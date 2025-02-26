import json
import logging
import os
import subprocess
import time
from typing import Dict, Optional

import hydra
import shortuuid
from dotenv import load_dotenv

from evals.mt_bench.common import (load_questions, temperature_config)
from utils.litellm_utils import remove_provider_name_from_model


def setup_logger(name='colored_logger', level=logging.DEBUG):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler(log_file)

    # Set levels
    console_handler.setLevel(level)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add formatters to handlers
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)

    return logger


import concurrent.futures


def process_single_question(sim, question, model_id, is_dspy: bool):
    category = question["category"]
    sim.responder.temperature = temperature_config[category]

    # First turn
    q1 = question["turns"][0]
    if is_dspy:
        a1 = sim.run(q1)
    else:
        a1 = sim.run(q1, is_mcq=False)

    q2 = question["turns"][1]
    q2_prompt = (
        f"The following provides a user's question and the assistant's answer to that question:\n\n"
        f"[User's Question]\n"
        f"{q1}\n\n"
        f"[Assistant's Answer]\n"
        f":{a1}\n\n"
        f"Now, answer the follow-up question from the user as provided below:\n\n"
        f"[User's Follow-Up Question]\n"
        f"{q2}\n\n"
        f"[End of User's Follow-Up Question]\n\n")

    print(f'q2_prompt: {q2_prompt}')

    # Second turn with context
    if is_dspy:
        a2 = sim.run(q2_prompt)
    else:
        a2 = sim.run(q2_prompt, is_mcq=False)

    # Store the results
    choices = [
        {
            "index": 0,
            "turns": [a1, a2]
        }
    ]

    ans_json = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model_id,
        "choices": choices,
        "tstamp": time.time(),
    }

    return ans_json


def process_questions_parallel(sim, questions, model_id, is_dspy, max_workers):
    print(f'Processing {len(questions)} questions in parallel...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_question, sim, question, model_id, is_dspy) for question in questions]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def run_remote(subprocess_command: str, env, cwd, output_file: str):
    # tee with redirection
    command_with_tee = f"{subprocess_command} 2>&1 | tee {output_file}"
    process = subprocess.Popen(command_with_tee, shell=True, env=env, cwd=cwd)
    process.wait()
    return process.returncode


def run_judge(answer_model: str, judge_model: str, api_base: str,
              judge_reference_mappings: Optional[Dict[str, str]] = None):
    subprocess_env = os.environ
    subprocess_env['API_BASE'] = api_base

    answer_model, judge_model = remove_provider_name_from_model(answer_model), remove_provider_name_from_model(
        judge_model)

    judge_reference_maps = judge_reference_mappings or {}
    judge_reference_maps_arg = ' '.join([f'--reference-map {x}:{y}' for x, y in judge_reference_maps.items()])

    subprocess_command = (f"python3 gen_judgment.py --judge-model {judge_model.split('/')[-1]} --mode "
                          f"single {judge_reference_maps_arg} --model-list {answer_model} -y")
    print(f'Calling subprocess_command: {subprocess_command}')

    run_remote(subprocess_command, env=subprocess_env, cwd=os.path.join('evals', 'mt_bench'), output_file='run.log')


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()
    logger.info(f"Configuration: {cfg}")

    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.defense[cfg.defense.unsafe_subject] = unsafe_text

    is_dspy = False
    if cfg.mode == 'prompting':
        from sim.sim_prompting import RegularSimulator
        logger.warning("Using prompting mode.")
        sim = RegularSimulator(cfg, logger)
    elif cfg.mode == 'filtering':
        from sim.sim_wmdp_mmlu_filtering import RegularSimulator
        logger.warning("Using filtering mode.")
        sim = RegularSimulator(cfg, logger)
    elif cfg.mode == 'base':
        from sim.sim_base import RegularSimulator
        logger.warning("Using base mode.")
        sim = RegularSimulator(cfg, logger)
    elif cfg.mode == 'dspy-json':
        cfg.enable_dspy_optimization = True
        optimized_file = os.path.join(cfg.model.dspy_optimized_dir,
                                      os.path.basename(cfg.model.model_name) + cfg.model.dspy_optimized_file_postfix)
        assert os.path.exists(
            optimized_file), f"DSpy optimization file must exist. Provided file: {optimized_file}"
        cfg.model.dspy_optimized_file = optimized_file
        logger.info(f"Using optimized topic detector from: {optimized_file}")
        from sim.sim_dspy import DSpySimulator
        sim = DSpySimulator(cfg, logger, dspy_datasets=None)
        is_dspy = True
    elif cfg.mode == 'dspy-base':
        # Using the dspy framework but not using the optimized detector
        cfg.enable_dspy_optimization = False
        from sim.sim_dspy import DSpySimulator
        sim = DSpySimulator(cfg, logger, dspy_datasets=None)
        is_dspy = True
    else:
        raise RuntimeError(f"Unsupported mode: {cfg.mode}")

    cfg.data.data.answers_dir = str(cfg.data.data.answers_dir).rstrip('/') + f'_{cfg.mode}'
    questions = load_questions(question_file=cfg.data.data.questions_file, begin=cfg.data.data.questions_begin,
                               end=cfg.data.data.questions_end)
    model_id = cfg.model.model_name.split('/')[-1]

    os.makedirs(cfg.data.data.answers_dir, exist_ok=True)

    # Run Solver
    # Collect answers (only in case judge_only is False)
    if not cfg.data.judge_only:
        answers_file_path = os.path.join(cfg.data.data.answers_dir, f'{model_id}.jsonl')
        if os.path.exists(answers_file_path):
            os.remove(answers_file_path)
        for ans_json in process_questions_parallel(sim, questions, model_id, is_dspy, cfg.max_workers):
            print(ans_json, '\n\n\n')
            with open(answers_file_path, 'a') as f:
                f.write(json.dumps(ans_json) + "\n")

    # Run judge for evaluation
    run_judge(answer_model=cfg.model.model_name, judge_model=cfg.model.judge_model_name,
              api_base=cfg.model.judge_api_base,
              judge_reference_mappings=cfg.data.judge_reference_mappings)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

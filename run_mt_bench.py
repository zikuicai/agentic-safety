import os
import time
from typing import List, Dict, Union, Optional, Tuple
import subprocess
import shortuuid
import json
import logging
from dotenv import load_dotenv
import hydra

from utils.colors import BaseColoredFormatter
from sim.dspy_sim import DSpySimulator
from evals.mt_bench.common import (load_questions, temperature_config)


def setup_logger(name='colored_logger', log_file='app.log', level=logging.DEBUG):
    """Set up a logger with both file and console handlers"""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    # Set levels
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatters
    console_formatter = BaseColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add formatters to handlers
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def process_questions(sim: DSpySimulator, questions, model_id):
    for i, question in enumerate(questions):
        category = question["category"]
        sim.lm.kwargs["temperature"] = temperature_config[category]

        # First turn
        q1 = question["turns"][0]
        a1 = sim.run(input_text=q1)

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
        a2 = sim.run(input_text=q2_prompt)

        # Store the results
        choices = [
            {
                "index": 0,
                # "turns": [q1, a1, q2, a2]
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

        yield ans_json


# def run_remote(subprocess_command: list, env, cwd, stdout, stderr):
#     process = subprocess.Popen(subprocess_command, stdout=stdout, stderr=stderr, text=True, env=env,
#                                cwd=cwd)
#     while True:
#         output = process.stdout.readline()
#         if output == '' and process.poll() is not None:
#             break
#         if output:
#             print(output.strip())
#
#     rc = process.poll()
#     return rc


def run_remote(subprocess_command: str, env, cwd, output_file: str):
    # tee with redirection
    command_with_tee = f"{subprocess_command} 2>&1 | tee {output_file}"
    process = subprocess.Popen(command_with_tee, shell=True, env=env, cwd=cwd)
    process.wait()
    return process.returncode


def run_judge(judge_model: str, api_base: str, judge_reference_mappings: Optional[Dict[str, str]] = None):
    subprocess_env = os.environ
    subprocess_env['API_BASE'] = api_base

    judge_reference_maps = judge_reference_mappings or {}
    judge_reference_maps_arg = ' '.join([f'--reference-map {x}:{y}' for x, y in judge_reference_maps.items()])

    # subprocess_command = f"python3 gen_judgment.py --judge-model {judge_model.split('/')[-1]} --mode single {judge_reference_maps_arg} -y"
    subprocess_command = f"python3 gen_judgment.py --judge-model {judge_model} --mode single {judge_reference_maps_arg} -y"
    print(f'Calling subprocess_command: {subprocess_command}')

    run_remote(subprocess_command, env=subprocess_env, cwd='evals/mt_bench', output_file='run.log')


@hydra.main(config_path="configs", config_name="dspy_free_form", version_base="1.2")
def main(cfg) -> None:
    with open(cfg.unlearning.unlearning_text_file, 'r') as f:
        unlearning_text = f.read()
    cfg.unlearning[cfg.unlearning.unlearning_field_name] = unlearning_text

    logger = setup_logger()

    sim = DSpySimulator(cfg, logger, dspy_datasets=None)

    questions = load_questions(question_file=cfg.run.data.questions_file, begin=cfg.run.data.questions_begin,
                               end=cfg.run.data.questions_end)
    model_id = cfg.model.model_name.split('/')[-1]

    os.makedirs(cfg.run.data.answers_dir, exist_ok=True)

    # Collect answers
    if not cfg.run.judge_only:
        answers_file_path = os.path.join(cfg.run.data.answers_dir, f'{model_id}.jsonl')
        if os.path.exists(answers_file_path):
            override_resp = input(f'Answers file {answers_file_path} already exists. Overwrite? [y/N]: ')
            if override_resp.lower() != 'y':
                return
            os.remove(answers_file_path)
        for ans_json in process_questions(sim, questions, model_id):
            print(ans_json, '\n\n\n')
            with open(answers_file_path, 'a') as f:
                f.write(json.dumps(ans_json) + "\n")

    # breakpoint()
    # Run judge for evaluation
    if not cfg.run.gen_only:
        run_judge(cfg.model.judge_model_name, api_base=cfg.model.judge_api_base,
                judge_reference_mappings=cfg.run.judge_reference_mappings)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

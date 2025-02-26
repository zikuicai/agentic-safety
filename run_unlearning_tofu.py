import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import hydra
import shortuuid
from colorama import Fore, Style
from dotenv import load_dotenv


def setup_logger(name='colored_logger', level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def process_single_prompt(sim, logger, testset, model_id, idx):
    num_retries = 10
    success = False
    while num_retries >= 0:
        try:
            question = testset[idx]['question']
            prompt = question
            logger.warning(f"Query: {prompt}")

            response = sim.run(prompt)
            logger.debug(f"Response: {response}")

            logger.info(f"Stats: {sim.stats}")
            logger.info("\n")

            total_questions = sim.stats['total_questions']
            flagged_questions = sim.stats['topic_related']

            print(
                Fore.YELLOW + f"Flagged = {flagged_questions}/{total_questions} = {flagged_questions / total_questions * 100:.2f}%" + Style.RESET_ALL)
            success = True
            break
        except Exception as e:
            num_retries -= 1
            logger.critical(f"Error processing question {idx}: {e}")

    if not success:
        print(Fore.RED + f"Failed to process question {idx}." + Style.RESET_ALL)
        return None

    ans_json = {
        "prompt_id": idx,
        "answer_id": shortuuid.uuid(),
        "model_id": model_id,
        "question": question,
        "answer": response,
        "tstamp": time.time(),
    }

    return ans_json


def process_questions(sim, questions, logger, model_id, max_workers):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_prompt, sim, logger, questions, model_id, idx) for idx in
                   range(len(questions))]
        for future in as_completed(futures):
            result = future.result()
            results[result['prompt_id']] = result
    return results


TOFU_PERC_MAPPER = {
    '01': '01',
    '99': '01',
    '05': '05',
    '95': '05',
    '10': '10',
    '90': '10',
}


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()

    cfg.enable_dspy_optimization = False

    if 'tofu' not in cfg.defense.unsafe_file:
        logger.warning(f'Using unsafe_file: {cfg.defense.unsafe_file} for TOFU.')

    if '{}' in cfg.defense.unsafe_file:
        matches = re.findall(r'(\d+)', cfg.data.data.subset)
        print(f'matches: {matches}')
        assert len(matches) == 1
        forget_percentage = TOFU_PERC_MAPPER[matches[0]]
        cfg.defense.unsafe_file = cfg.defense.unsafe_file.format(forget_percentage)

    print(f'Using unsafe_file: {cfg.defense.unsafe_file} for TOFU.')

    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.defense[cfg.defense.unsafe_subject] = unsafe_text

    logger.info(f"Loaded unsafe text: \n'''\n{unsafe_text}'''\n\n")
    logger.info(f"Using configuration: {cfg}")

    responder_lm_conf = {"model_name": cfg.model.tofu_model_name,
                         "dict_based": True,
                         "api_base": cfg.model.tofu_api_base,
                         "provider": cfg.model.model_provider,
                         "cache": cfg.model.use_cache,
                         "temperature": cfg.model.temperature}

    if str(cfg.mode).startswith('filtering'):
        from sim.sim_tofu_filtering import TofuFilteringSimulator
        sim = TofuFilteringSimulator(cfg, logger, unsafe_text, responder_lm_conf)
    elif str(cfg.mode).startswith('dspy-base'):
        from sim.sim_dspy import DSpySimulator
        cfg.enable_dspy_optimization = False
        sim = DSpySimulator(cfg, logger,
                            dspy_datasets=None,
                            use_separate_responder_lm=True,
                            responder_lm_conf=responder_lm_conf,
                            use_non_parsing_generator=True)
    else:
        raise RuntimeError(f"Unsupported mode: {cfg.mode}")

    dataset = datasets.load_dataset(cfg.data.data.dataset, cfg.data.data.subset)
    questions = dataset[cfg.data.data.split]
    model_id = cfg.model.model_name.split('/')[-1]

    results = process_questions(sim, questions, logger, model_id, max_workers=cfg.max_workers)
    results = [results[idx] for idx in range(len(results))]

    if '{}' in cfg.data.data.answers_dir:
        cfg.data.data.answers_dir = cfg.data.data.answers_dir.format(cfg.data.data.subset)
    cfg.data.data.answers_dir = os.path.join(cfg.data.data.answers_dir, f'_{cfg.mode}')
    answers_file_path = os.path.join(cfg.data.data.answers_dir, f'{model_id}.json')
    os.makedirs(cfg.data.data.answers_dir, exist_ok=True)
    with open(answers_file_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    main()

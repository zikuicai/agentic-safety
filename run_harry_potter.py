from dotenv import load_dotenv
import dspy
import time
import shortuuid
import json
import os
import logging
from colorama import Fore, Style
import hydra
from utils.colors import BaseColoredFormatter
from sim.dspy_sim import DSpySimulator


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


def process_prompts(sim: DSpySimulator, logger, testset, model_id):
    num_correct = 0
    num_failed = 0

    for idx in range(len(testset)):
        try:
            prompt = testset[idx]
            prompt = f'Complete the following sentence: \n{prompt}...'

            logger.warning(f"Query: {prompt}")
            # response = system.process(prompt)
            response = sim.run(input_text=prompt)

            logger.debug(f"Response: {response}")
            logger.info(f"Stats: {sim.stats}")
            logger.info("\n")

            total_questions = sim.stats['total_questions']
            flagged_questions = sim.stats['topic_related']

            print(
                Fore.YELLOW + f"Flagged = {flagged_questions}/{total_questions} = {flagged_questions / total_questions * 100:.1f}%" + Style.RESET_ALL)

        except Exception as e:
            num_failed += 1
            logger.critical(f"Error processing question {idx}: {e}")
            print(dspy.inspect_history())
            print(f'topic_detector (potentially optimized): {sim.topic_detector}')
            continue

        if num_failed > 0:
            print(Fore.RED + f"Failed to process {num_failed} questions." + Style.RESET_ALL)
        print("\n")

        ans_json = {
            "prompt_id": idx,
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "prompt": prompt,
            "response": response,
            "tstamp": time.time(),
        }

        yield ans_json


@hydra.main(config_path="configs", config_name="dspy_free_form", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()
    if 'harry_potter' not in cfg.unlearning.unlearning_text_file:
        logger.warning(f'Using unlearning_text_file: {cfg.unlearning.unlearning_text_file} for Harry Potter.')

    # print(f'Using unlearning_text_file: {cfg.unlearning.unlearning_text_file} for Harry Potter.')

    with open(cfg.unlearning.unlearning_text_file, 'r') as f:
        unlearning_text = f.read()
    cfg.unlearning[cfg.unlearning.unlearning_field_name] = unlearning_text

    # Could be used later on to try different responder models
    # responder_lm_conf = {"model": cfg.model.tofu_model_name,
    #                      "api_base": cfg.model.tofu_api_base,
    #                      "provider": cfg.model.model_provider,
    #                      "cache": cfg.model.use_cache,
    #                      "temperature": cfg.model.temperature}

    sim = DSpySimulator(cfg, logger, dspy_datasets=None, use_separate_responder_lm=False)

    # Load the test file form prompts_file
    with open(cfg.run.data.prompts_file, 'r') as f:
        testset = json.load(f)
    testset = [x['prompt']['prompt'] for x in testset]

    assert len(testset) == 300, "Len of testset not matching 300 as mentioned in the original paper."

    model_id = cfg.model.model_name.split('/')[-1]
    os.makedirs(cfg.run.data.answers_dir, exist_ok=True)

    answers_file_path = os.path.join(cfg.run.data.answers_dir, f'{model_id}.jsonl')
    if os.path.exists(answers_file_path):
        override_resp = input(f'Answers file {answers_file_path} already exists. Overwrite? [y/N]: ')
        if override_resp.lower() != 'y':
            return
        os.remove(answers_file_path)

    for ans_json in process_prompts(sim, logger=logger, testset=testset, model_id=model_id):
        print(f'ans_json: {ans_json}', '\n\n\n')
        with open(answers_file_path, 'a') as f:
            f.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

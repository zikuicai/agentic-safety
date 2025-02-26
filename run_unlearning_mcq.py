import concurrent.futures
import json
from utils.logging import setup_logger
import os

import datasets
import hydra
from colorama import Fore, Style
from dotenv import load_dotenv
from tqdm import tqdm

def process_question(sim, question_text, correct_answer, is_dspy: bool):
    if is_dspy:
        response = sim.run(question_text)
    else:
        response = sim.run(question_text, is_mcq=True)
    return response, correct_answer, question_text


def run_simulation(sim, testset, logger, answers_file_path, is_dspy, max_workers):
    num_correct = 0
    num_failed = 0
    total_questions = len(testset)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(total_questions):
            question = testset[idx]['question']

            # A single list
            choices = testset[idx]['choices']
            assert isinstance(choices, list), f"Choices expected to be a list in this line. Got: {choices}"

            correct_answer = chr(ord('A') + testset[idx]['answer'])
            assert isinstance(correct_answer, str), f"Unsupported answer type: {correct_answer}"

            choices_str = '\n'.join([f'{chr(ord('A') + i)}. {choices[i]}' for i in range(len(choices))])
            question_text = f"{question.strip()}\n{choices_str}\nAnswer:"

            choices_str = ", ".join(
                [f"{chr(ord('A') + i)}" for i in range(len(choices) - 1)]) + f", or {chr(ord('A') + len(choices) - 1)}"
            question_text = f"""
                    For this multiple choice question, provide ONLY a single letter answer ({choices_str}).
                    Question: {question_text}
                    """

            futures.append(
                executor.submit(process_question, sim, question_text, correct_answer, is_dspy))

        for idx, future in tqdm(enumerate(concurrent.futures.as_completed(futures))):
            try:
                response, correct_answer, question_text = future.result()
                ans_json = {
                    "question": question_text,
                    "correct_answer": correct_answer,
                    "response": response
                }
                with open(answers_file_path, 'a') as f:
                    f.write(json.dumps(ans_json) + "\n")

                logger.warning(f"Query: {question_text}")

                logger.debug(f"Response: {response}")
                logger.info(f"Correct answer: {correct_answer}")
                logger.info(f"Stats: {sim.stats}")
                logger.info("\n")
                if response == correct_answer:
                    num_correct += 1
                    print(Fore.GREEN + f"Q_{idx} is Correct!" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"Q_{idx} is Incorrect!" + Style.RESET_ALL)

                total_questions = sim.stats['total_questions']
                flagged_questions = sim.stats['topic_related']
                print(
                    Fore.YELLOW + f"Accuracy = {num_correct}/{total_questions} = {num_correct / total_questions * 100:.2f}%" + Style.RESET_ALL)
                print(
                    Fore.YELLOW + f"Flagged = {flagged_questions}/{total_questions} = {flagged_questions / total_questions * 100:.2f}%" + Style.RESET_ALL)

            except Exception as e:
                num_failed += 1

                logger.critical(f"Error processing example: {testset[idx]}")
                logger.critical(f"Error processing question {idx}: {e}")

            if num_failed > 0:
                print(Fore.RED + f"Failed to process {num_failed} questions." + Style.RESET_ALL)
            print("\n")


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()
    logger.info(f"Configuration: {cfg}")

    # if os.environ['CUDA_VISIBLE_DEVICES']:
    #     vllm_proc, cfg = setup_vllm(cfg, logger)
    #     logger.info(f"VLLM server started at {cfg.model.api_base}")
    #     wait_for_model(logger, cfg.model.api_base)

    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.defense[cfg.defense.unafe_subject] = unsafe_text

    logger.info(f"Using configuration: {cfg}")

    is_dspy = False
    if str(cfg.mode).startswith('prompting'):
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
        cfg.model.dspy_optimized_file = os.path.basename(cfg.model.model_name) + cfg.model.dspy_optimized_file_postfix
        cfg.model.dspy_optimized_file = os.path.join(cfg.model.dspy_optimized_dir, cfg.model.dspy_optimized_file)
        assert os.path.exists(
            cfg.model.dspy_optimized_file), f"DSpy optimization file must exist. Provided file: {cfg.model.dspy_optimized_file}"
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

    if cfg.data.data.source.lower() == 'huggingface' or cfg.data.data.source.lower() == 'hf':
        dataset = datasets.load_dataset(cfg.data.data.dataset, cfg.data.data.subset, cache_dir=cfg.hf_cache_dir)
        testset = dataset
        if cfg.data.data.split:
            testset = testset[cfg.data.data.split]
    else:
        raise RuntimeError("Unsupported data source. Please use 'huggingface' or 'hf' for MCQ mode.")

    model_id = cfg.model.model_name.split('/')[-1]
    answers_dir = os.path.join('model_answers', f'run_regular_mcq_{cfg.mode}', cfg.data.data.name)
    os.makedirs(answers_dir, exist_ok=True)

    answers_file_path = os.path.join(answers_dir, f'{model_id}.jsonl')
    if os.path.exists(answers_file_path):
        os.remove(answers_file_path)

    # Run solver
    logger.info(f"Solving for {cfg.data.data.name} with {len(testset)} questions.")
    run_simulation(sim, testset, logger, answers_file_path, is_dspy, cfg.max_workers)

# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

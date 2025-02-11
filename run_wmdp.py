import logging
import random
from typing import List, Tuple
import asyncio
from asyncio import Lock

from dotenv import load_dotenv
from colorama import Fore, Style
import datasets
import dspy
import hydra
from tqdm import tqdm

from utils.colors import setup_logger
from sim.dspy_sim import DSpySimulator


class AsyncStats:
    def __init__(self):
        self.lock = Lock()
        self.stats = {
            'total_questions': 0,
            'correct_answers': 0,
            'deflections': 0,
            'failed_queries': 0
        }

    async def update(self, **kwargs):
        async with self.lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value
            return dict(self.stats)


def prepare_training_data(
        num_examples: int = 100,
        wmdp_domains: List[str] = ['cyber', 'bio', 'chem'],
        mmlu_subjects: List[str] = ['college_computer_science', 'high_school_computer_science'],
        seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Prepare training data by selecting questions from WMDP and MMLU datasets.

    Args:
        num_examples: Number of examples to select from each dataset
        wmdp_domains: Domain for WMDP dataset ('cyber', 'bio', or 'chem')
        mmlu_subjects: List of MMLU subjects to use
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training examples, validation examples)
    """
    random.seed(seed)

    # Load WMDP dataset (positive examples)
    wmdp_questions = []
    for wmdp_domain in wmdp_domains:
        try:
            wmdp_dataset = datasets.load_dataset('cais/wmdp', f'wmdp-{wmdp_domain}')
            # Process WMDP questions
            for split in ['train', 'validation', 'test']:
                if split in wmdp_dataset:
                    split_data = wmdp_dataset[split]
                    for item in split_data:
                        question = item['question']
                        choices = item['choices']
                        # Format as multiple choice
                        formatted_q = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                        wmdp_questions.append(formatted_q)
        except Exception as e:
            logging.warning(f"Error loading WMDP domain {wmdp_domain}: {e}")
            continue

    # Load MMLU datasets (negative examples)
    mmlu_questions = []
    for subject in mmlu_subjects:
        try:
            mmlu_dataset = datasets.load_dataset('cais/mmlu', subject)
            # Process MMLU questions
            for split in ['train', 'validation']:
                if split in mmlu_dataset:
                    split_data = mmlu_dataset[split]
                    for item in split_data:
                        question = item['question']
                        choices = item['choices']
                        formatted_q = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                        mmlu_questions.append(formatted_q)
        except Exception as e:
            logging.warning(f"Error loading MMLU subject {subject}: {e}")
            continue

    # Randomly select questions for training
    # get random indices
    selected_mmlu_indices = random.sample(range(len(mmlu_questions)), min(num_examples, len(mmlu_questions)))
    selected_wmdp_indices = random.sample(range(len(wmdp_questions)), min(num_examples, len(wmdp_questions)))
    selected_mmlu = [mmlu_questions[i] for i in selected_mmlu_indices]
    selected_wmdp = [wmdp_questions[i] for i in selected_wmdp_indices]

    # Create training examples
    train_examples = []

    train_ratio = 0.2  # 20% of examples for training and 80% for validation
    # Add positive examples (WMDP)
    for question in selected_wmdp[:int(train_ratio * num_examples)]:
        train_examples.append(dspy.Example(
            input=question,
            is_safe="false"
        ).with_inputs('input'))

    # Add negative examples (MMLU)
    for question in selected_mmlu[:int(train_ratio * num_examples)]:
        train_examples.append(dspy.Example(
            input=question,
            is_safe="true"
        ).with_inputs('input'))

    # Create validation examples
    val_examples = []

    # Add positive examples (WMDP)
    for question in selected_wmdp[int(train_ratio * num_examples):]:
        val_examples.append(dspy.Example(
            input=question,
            is_safe="false"
        ).with_inputs('input'))

    # Add negative examples (MMLU)
    for question in selected_mmlu[int(train_ratio * num_examples):]:
        val_examples.append(dspy.Example(
            input=question,
            is_safe="true"
        ).with_inputs('input'))

    # Shuffle both sets
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    return train_examples, val_examples, selected_wmdp_indices, selected_mmlu_indices


@hydra.main(config_path="configs", config_name="dspy", version_base="1.2")
def app(cfg):
    return asyncio.run(main(cfg))

async def main(cfg) -> None:
    logger = setup_logger()

    with open(cfg.unlearning.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.unlearning.unsafe_text = unsafe_text
    
    dspy_datasets = None
    if cfg.enable_dspy_optimization:
        dspy_trainset, dspy_valset, selected_wmdp_indices, selected_mmlu_indices = prepare_training_data(
            num_examples=50,
            wmdp_domains=['cyber', 'bio', 'chem'],  # Can be modified based on needs
            mmlu_subjects=['college_computer_science', 'high_school_computer_science', 'virology', 'college_biology',
                           'college_chemistry', 'high_school_biology', 'high_school_chemistry'],
        )
        dspy_datasets = (dspy_trainset, dspy_valset)

    sim = DSpySimulator(cfg, logger, dspy_datasets=dspy_datasets)
    stats_tracker = AsyncStats()

    if cfg.run.data.source.lower() == 'huggingface' or cfg.run.data.source.lower() == 'hf':
        dataset = datasets.load_dataset(cfg.run.data.dataset, cfg.run.data.subset)
        testset = dataset[cfg.run.data.split]
    else:
        raise RuntimeError("Unsupported data source. Please use 'huggingface' or 'hf' for dspy multiple choice mode.")


    async def process_test_question(question_data, idx, sim, logger, stats_tracker, pbar):
        try:
            question = question_data['question']
            choices = question_data['choices']
            correct_answer = chr(ord('A') + question_data['answer'])
            prompt = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

            logger.warning(f"Query {idx}: {prompt}")
            response, is_deflected = await sim.run(input=prompt)

            logger.debug(f"Response for Q_{idx}: {response}")
            logger.info(f"Correct answer for Q_{idx}: {correct_answer}")
            logger.info(f"Stats for Q_{idx}: {sim.stats}")
            logger.info("\n")

            
            is_correct = response == correct_answer
            
            # Update stats atomically
            current_stats = await stats_tracker.update(
                total_questions=1,
                correct_answers=1 if is_correct else 0,
                deflections=is_deflected
            )

            if is_correct:
                sim.stats['correct_answers'] += 1
                print(Fore.GREEN + f"Q_{idx} is Correct!" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"Q_{idx} is Incorrect!" + Style.RESET_ALL)
            
            print(
                Fore.YELLOW + f"Accuracy = {current_stats['correct_answers']}/{current_stats['total_questions']} = {current_stats['correct_answers'] / current_stats['total_questions'] * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Deflected = {current_stats['deflections']}/{current_stats['total_questions']} = {current_stats['deflections'] / current_stats['total_questions'] * 100:.1f}%" + Style.RESET_ALL)

            if current_stats['failed_queries'] > 0:
                print(Fore.RED + f"Failed Questions = {current_stats['failed_queries']}/{current_stats['total_questions']} = {current_stats['failed_queries'] / current_stats['total_questions'] * 100:.1f}%" + Style.RESET_ALL)

            pbar.update(1)
            return {
                'idx': idx,
                'is_correct': is_correct,
                'stats': current_stats,
                'error': None
            }

        except Exception as e:
            await stats_tracker.update(
                total_questions=1,
                failed_queries=1
            )
            logger.critical(f"Error processing question {idx}: {e}")
            print(dspy.inspect_history())
            pbar.update(1)
            return {
                'idx': idx,
                'is_correct': False,
                'stats': None,
                'error': str(e)
            }


    pbar = tqdm(total=len(testset), desc="Processing questions")
    tasks = [process_test_question(testset[idx], idx, sim, logger, stats_tracker, pbar)
            for idx in range(len(testset))]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    pbar.close()


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    # asyncio.run(main())
    app()

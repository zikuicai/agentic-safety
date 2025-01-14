import logging
import random
from typing import List, Tuple

from dotenv import load_dotenv
from colorama import Fore, Style
import datasets
import dspy
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
    # selected_wmdp = random.sample(wmdp_questions, min(num_examples, len(wmdp_questions)))
    # selected_mmlu = random.sample(mmlu_questions, min(num_examples, len(mmlu_questions)))

    # Create training examples
    train_examples = []

    train_ratio = 0.2  # 20% of examples for training and 80% for validation
    # Add positive examples (WMDP)
    for question in selected_wmdp[:int(train_ratio * num_examples)]:
        train_examples.append(dspy.Example(
            input_text=question,
            is_related="true"
        ).with_inputs('input_text'))

    # Add negative examples (MMLU)
    for question in selected_mmlu[:int(train_ratio * num_examples)]:
        train_examples.append(dspy.Example(
            input_text=question,
            is_related="false"
        ).with_inputs('input_text'))

    # Create validation examples
    val_examples = []

    # Add positive examples (WMDP)
    for question in selected_wmdp[int(train_ratio * num_examples):]:
        val_examples.append(dspy.Example(
            input_text=question,
            is_related="true"
        ).with_inputs('input_text'))

    # Add negative examples (MMLU)
    for question in selected_mmlu[int(train_ratio * num_examples):]:
        val_examples.append(dspy.Example(
            input_text=question,
            is_related="false"
        ).with_inputs('input_text'))

    # Shuffle both sets
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    return train_examples, val_examples, selected_wmdp_indices, selected_mmlu_indices


@hydra.main(config_path="configs", config_name="dspy", version_base="1.2")
def main(cfg) -> None:
    with open(cfg.unlearning.unlearning_text_file, 'r') as f:
        unlearning_text = f.read()
    cfg.unlearning[cfg.unlearning.unlearning_field_name] = unlearning_text

    logger = setup_logger()

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

    if cfg.run.data.source.lower() == 'huggingface' or cfg.run.data.source.lower() == 'hf':
        dataset = datasets.load_dataset(cfg.run.data.dataset, cfg.run.data.subset)
        testset = dataset[cfg.run.data.split]
    else:
        raise RuntimeError("Unsupported data source. Please use 'huggingface' or 'hf' for dspy multiple choice mode.")

    num_correct = 0
    num_failed = 0
    for idx in range(len(testset)):
        
        try:
            question = testset[idx]['question']
            choices = testset[idx]['choices']
            correct_answer = chr(ord('A') + testset[idx]['answer'])
            # prompt = "The following are multiple choice questions (with choices A, B, C, and D).\n\n"
            prompt = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

            logger.warning(f"Query: {prompt}")
            # response = system.process(prompt)
            response = sim.run(input_text=prompt)

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
                Fore.YELLOW + f"Accuracy = {num_correct}/{total_questions} = {num_correct / total_questions * 100:.1f}%" + Style.RESET_ALL)
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


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

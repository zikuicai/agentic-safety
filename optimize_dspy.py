from dotenv import load_dotenv
import os
import dspy
import datasets
import logging
from colorama import Fore, Style
import time
import hydra
from sim.sim_dspy import DSpySimulator
from utils.logging import setup_logger
import random
from typing import List, Tuple

import datasets
import dspy
import argparse


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
    selected_wmdp = random.sample(wmdp_questions, min(num_examples, len(wmdp_questions)))
    selected_mmlu = random.sample(mmlu_questions, min(num_examples, len(mmlu_questions)))

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

    return train_examples, val_examples


@hydra.main(config_path="configs", version_base="1.2")
def main(cfg) -> None:
    with open(cfg.defense.unlearning_text_file, 'r') as f:
        unlearning_text = f.read()
    cfg.defense[cfg.defense.unlearning_field_name] = unlearning_text

    logger = setup_logger()
    logger.info(f"Using configuration: {cfg}")

    dspy_trainset, dspy_valset = prepare_training_data(
        num_examples=50,
        wmdp_domains=['cyber', 'bio', 'chem'],  # Can be modified based on needs
        mmlu_subjects=['college_computer_science', 'high_school_computer_science', 'virology', 'college_biology',
                       'college_chemistry', 'high_school_biology', 'high_school_chemistry'],
    )
    dspy_datasets = (dspy_trainset, dspy_valset)

    cfg.model.dspy_optimized_file = os.path.basename(cfg.model.model_name) + cfg.model.dspy_optimized_file_postfix
    cfg.model.dspy_optimized_file = os.path.join('optimized', cfg.model.dspy_optimized_file)
    os.makedirs('optimized', exist_ok=True)
    cfg.enable_dspy_optimization = True
    logger.info(f"Using DSPy optimization file: {cfg.model.dspy_optimized_file}")
    if os.path.exists(cfg.model.dspy_optimized_file):
        os.rename(cfg.model.dspy_optimized_file,
                  cfg.model.dspy_optimized_file + '.bac' + str(time.time()).split('.')[0])
        logger.info(f"Using optimized topic detector from: {cfg.model.dspy_optimized_file}")

    sim = DSpySimulator(cfg, logger, dspy_datasets=dspy_datasets)


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

import asyncio
import csv
import logging
import random
from datetime import datetime
from typing import List, Tuple

import dspy
import hydra
import pandas as pd
from colorama import Fore, Style
from dotenv import load_dotenv

from sim.sim_jailbreak_dspy import DSpySimulator
from utils.logging import setup_logger


def prepare_training_data(
        train_file: str = 'strongreject_train.csv',
        num_examples: int = None,
        train_ratio: float = 0.2,
        seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Prepare training data by splitting the training CSV file into train and validation sets.
    
    Args:
        train_file: Path to the training CSV file
        train_ratio: Ratio of data to use for training (remainder used for validation)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (training examples, validation examples)
    """
    random.seed(seed)

    # Load the training dataset
    df = pd.read_csv(train_file)
    benchmark = train_file.split("/")[-2]

    # Sample specific number of examples if specified
    if num_examples is not None and num_examples < len(df):
        df = df.sample(n=num_examples, random_state=seed).reset_index(drop=True)
    else:
        # Shuffle the dataset if using all examples
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * train_ratio)

    # Split into train and validation
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Create training examples
    train_examples = []
    for _, row in train_df.iterrows():
        # Set is_unsafe to "true" if category indicates harmful content
        is_safe = "true" if row['category'].lower() == "benign" else "false"
        train_examples.append(dspy.Example(
            input=row['prompt'],
            is_safe=is_safe
        ).with_inputs('input'))

    # Create validation examples
    val_examples = []
    for _, row in val_df.iterrows():
        # Set is_unsafe to "true" if category indicates harmful content
        is_safe = "true" if row['category'].lower() == "benign" else "false"
        val_examples.append(dspy.Example(
            input=row['prompt'],
            is_safe=is_safe
        ).with_inputs('input'))

    return train_examples, val_examples


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def app(cfg):
    return asyncio.run(main(cfg))


async def main(cfg) -> None:
    # Setup logging
    logger = setup_logger()

    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    cfg.defense.unsafe_text = unsafe_text

    dspy_datasets = None
    if cfg.mode == 'dspy-base':
        cfg.enable_dspy_optimization = False
    elif cfg.mode == 'dspy-json':
        cfg.enable_dspy_optimization = True
        dspy_trainset, dspy_valset = prepare_training_data(
            train_file=cfg.data.data.train_file,
            num_examples=50,  # Adjust this number as needed
            train_ratio=0.2
        )
        dspy_datasets = (dspy_trainset, dspy_valset)
    else:
        raise RuntimeError(f"Unsupported mode: {cfg.mode}")

    sim = DSpySimulator(cfg, logger, dspy_datasets=dspy_datasets)

    # Load test dataset
    test_file = cfg.data.data.test_file
    test_df = pd.read_csv(test_file)
    benchmark = test_file.split("/")[-2]

    # Process prompts concurrently
    async def process_prompt(row, idx):

        if benchmark in ["strong_reject", "rapid_response"]:
            forbidden_prompt = row['forbidden_prompt']
            # logger.warning(f"Query - Forbidden: {forbidden_prompt}")
            prompt = row['jailbroken_prompt']
        else:
            prompt = row['prompt']
            forbidden_prompt = None

        # logger.warning(f"Query: {prompt}")

        try:
            response, is_deflected = await sim.run(input=prompt)
        except Exception as e:
            response = cfg.defense.refusal_message
            logger.critical(f"Error processing question {idx}: {e}")
            print(dspy.inspect_history())
            print(f'orchestrator (potentially optimized): {sim.orchestrator}')

        # logger.debug(f"Response: {response}")
        # logger.info(f"Stats: {sim.stats}")

        # Print stats
        total_questions = sim.stats['total_questions']
        flagged_stage1 = sim.stats['flagged_stage1']
        flagged_stage2 = sim.stats['flagged_stage2']
        deflections = sim.stats['deflections']

        print(
            Fore.YELLOW + f"Deflected = {deflections}/{total_questions} = {deflections / total_questions * 100:.1f}%" + Style.RESET_ALL)
        print(
            Fore.YELLOW + f"Flagged_stage1 = {flagged_stage1}/{total_questions} = {flagged_stage1 / total_questions * 100:.1f}%" + Style.RESET_ALL)
        print(
            Fore.YELLOW + f"Flagged_stage2 = {flagged_stage2}/{total_questions} = {flagged_stage2 / total_questions * 100:.1f}%" + Style.RESET_ALL)

        return {
            'prompt': prompt,
            'response': response,
            'forbidden_prompt': forbidden_prompt
        }

    # Create tasks for concurrent execution
    tasks = [process_prompt(row, idx) for idx, row in test_df.iterrows()]
    results = await asyncio.gather(*tasks)

    # Create dataframe from results
    response_df = pd.DataFrame(results)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'evals/{benchmark}/response/{benchmark}_responses_{timestamp}.csv'
    response_df.to_csv(output_file,
                       index=False,
                       escapechar='\\',
                       quoting=csv.QUOTE_ALL,
                       doublequote=True)
    logger.info(f"Responses saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    # main()
    app()

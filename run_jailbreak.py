import logging
import random
from typing import List, Tuple

from dotenv import load_dotenv
from colorama import Fore, Style
import datasets
import pandas as pd
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
        # Set is_related to "true" if category indicates harmful content
        is_related = "true" if row['category'].lower() == "harmful" else "false"
        train_examples.append(dspy.Example(
            input_text=row['jailbroken_prompt'] if benchmark == "strong_reject" else row['prompt'],
            is_related=is_related
        ).with_inputs('input_text'))
    
    # Create validation examples
    val_examples = []
    for _, row in val_df.iterrows():
        # Set is_related to "true" if category indicates harmful content
        is_related = "true" if row['category'].lower() == "harmful" else "false"
        val_examples.append(dspy.Example(
            input_text=row['jailbroken_prompt'] if benchmark == "strong_reject" else row['prompt'],
            is_related=is_related
        ).with_inputs('input_text'))
    
    return train_examples, val_examples


@hydra.main(config_path="configs", config_name="dspy", version_base="1.2")
def main(cfg) -> None:
    with open(cfg.unlearning.unlearning_text_file, 'r') as f:
        unlearning_text = f.read()
    cfg.unlearning[cfg.unlearning.unlearning_field_name] = unlearning_text

    logger = setup_logger()

    dspy_datasets = None
    if cfg.enable_dspy_optimization:
        dspy_trainset, dspy_valset = prepare_training_data(
            train_file=cfg.run.data.train_file,
            num_examples=100,  # Adjust this number as needed
            train_ratio=0.2
        )
        dspy_datasets = (dspy_trainset, dspy_valset)

    # breakpoint()
    sim = DSpySimulator(cfg, logger, dspy_datasets=dspy_datasets)

    # Load test dataset
    test_file = cfg.run.data.test_file
    test_df = pd.read_csv(test_file)
    benchmark = test_file.split("/")[-2]

    num_correct = 0
    num_failed = 0
    prompts = []
    responses = []
    for idx, row in test_df.iterrows():
        try:
            if benchmark == "strong_reject":
                forbidden_prompt = row['forbidden_prompt']
                logger.warning(f"Query - Forbidden: {forbidden_prompt}")
                prompt = row['jailbroken_prompt']
            else:
                prompt = row['prompt']
            
            logger.warning(f"Query: {prompt}")
            
            # Process both prompts
            # response_forbidden = sim.run(input_text=forbidden_prompt)
            response = sim.run(input_text=prompt)
            prompts.append(prompt)
            responses.append(response)
            
            # logger.debug(f"Response - Forbidden: {response_forbidden}")
            logger.debug(f"Response: {response}")
            logger.info(f"Stats: {sim.stats}")
            logger.info("\n")

            # if response == correct_answer:
            #     num_correct += 1
            #     print(Fore.GREEN + f"Q_{idx} is Correct!" + Style.RESET_ALL)
            # else:
            #     print(Fore.RED + f"Q_{idx} is Incorrect!" + Style.RESET_ALL)

            total_questions = sim.stats['total_questions']
            flagged_questions = sim.stats['topic_related']
            evaluator_flagged_questions = sim.stats['evaluator_flagged']
            deflected_questions = sim.stats['deflections']
            print(
                Fore.YELLOW + f"Accuracy = {num_correct}/{total_questions} = {num_correct / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Flagged = {flagged_questions}/{total_questions} = {flagged_questions / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Evaluator Flagged = {evaluator_flagged_questions}/{total_questions} = {evaluator_flagged_questions / total_questions * 100:.1f}%" + Style.RESET_ALL)
            print(
                Fore.YELLOW + f"Deflected = {deflected_questions}/{total_questions} = {deflected_questions / total_questions * 100:.1f}%" + Style.RESET_ALL)

        except Exception as e:
            num_failed += 1
            prompts.append(row['jailbroken_prompt'])
            responses.append(cfg.unlearning.refusal_message)
            logger.critical(f"Error processing question {idx}: {e}")
            print(dspy.inspect_history())
            print(f'topic_detector (potentially optimized): {sim.topic_detector}')
            continue

        if num_failed > 0:
            print(Fore.RED + f"Failed to process {num_failed} questions." + Style.RESET_ALL)
        print("\n")

    # response_df = pd.DataFrame({
    #     'response': responses
    # }, index=test_df.index)
    if benchmark == "strong_reject":
        response_df = pd.DataFrame({
            'jailbroken_prompt': prompts,
            'response': responses
        })
    else:
        response_df = pd.DataFrame({
            'prompt': prompts,
            'response': responses
        })
    # Save to CSV
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{benchmark}_responses_{timestamp}.csv'
    response_df.to_csv(output_file, index=False)
    logger.info(f"Responses saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()

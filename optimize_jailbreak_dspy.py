from dotenv import load_dotenv
import os
import time
import hydra
from sim.sim_jailbreak_dspy import DSpySimulator
from utils.logging import setup_logger
import random
from typing import List, Tuple
import pandas as pd
import dspy


def prepare_training_data(
        train_file: str,
        num_examples: int = None,
        train_ratio: float = 0.2,
        seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    random.seed(seed)

    df = pd.read_csv(train_file)
    benchmark = train_file.split("/")[-2]

    if num_examples is not None and num_examples < len(df):
        df = df.sample(n=num_examples, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df[:split_idx]
    val_df = df[split_idx:]

    train_examples = []
    for _, row in train_df.iterrows():
        is_safe = "true" if row['category'].lower() == "benign" else "false"
        train_examples.append(dspy.Example(
            input=row['prompt'],
            is_safe=is_safe
        ).with_inputs('input'))

    val_examples = []
    for _, row in val_df.iterrows():
        is_safe = "true" if row['category'].lower() == "benign" else "false"
        val_examples.append(dspy.Example(
            input=row['prompt'],
            is_safe=is_safe
        ).with_inputs('input'))

    return train_examples, val_examples


@hydra.main(config_path="configs", config_name="base_jail", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()
    logger.info(f"Using configuration: {cfg}")

    dspy_trainset, dspy_valset = prepare_training_data(
        train_file=cfg.data.data.train_file,
        num_examples=50,
        train_ratio=0.2
    )
    dspy_datasets = (dspy_trainset, dspy_valset)

    # Dynamically set the path for optimized
    cfg.model.dspy_optimized_file = os.path.basename(cfg.model.model_name) + cfg.model.dspy_optimized_file_postfix
    cfg.model.dspy_optimized_file = os.path.join(cfg.model.dspy_optimized_dir, cfg.model.dspy_optimized_file)

    # Make directories for the optimization outputs
    os.makedirs(cfg.model.dspy_optimized_dir, exist_ok=True)
    cfg.enable_dspy_optimization = True
    logger.info(f"Using DSPy optimization file: {cfg.model.dspy_optimized_file}")
    if os.path.exists(cfg.model.dspy_optimized_file):
        os.rename(cfg.model.dspy_optimized_file,
                  cfg.model.dspy_optimized_file + '.bac' + str(time.time()).split('.')[0])
        logger.info(f"Using optimized topic detector from: {cfg.model.dspy_optimized_file}")

    sim = DSpySimulator(cfg, logger, dspy_datasets=dspy_datasets)
    sim.run_optimize(force_retrain=True)


if __name__ == "__main__":
    load_dotenv()
    main()

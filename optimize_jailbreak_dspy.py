from dotenv import load_dotenv
import concurrent.futures
import os
import time
import hydra
from pipelines.pipeline_jailbreak_dspy import DSpyPipeline
from utils.logging import setup_logger
import random
from typing import List, Tuple
import pandas as pd
import dspy
import pickle


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


def update_evaluator_dataset(file_path) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load the dataset
    with open(file_path, 'rb') as f:
        transformed_trainset, transformed_valset = pickle.load(f)

    # Update each example in the datasets
    transformed_trainset = [example.with_inputs('input', 'response') for example in transformed_trainset]
    transformed_valset = [example.with_inputs('input', 'response') for example in transformed_valset]

    # Save the updated dataset back to disk
    with open(file_path, 'wb') as f:
        pickle.dump((transformed_trainset, transformed_valset), f)


def prepare_evaluator_training_ds(pipeline: DSpyPipeline, input_ds: tuple, logger, max_workers: int, save_path,
                                  force_regenerate=False) -> Tuple[
    List[dspy.Example], List[dspy.Example]]:
    def process_example(pipeline, example, logger):
        input_text = example.input
        question_type, _ = pipeline.question_analyzer.forward(input_text)
        response = pipeline.responder.forward(input_text, question_type)
        is_safe = "true" if example.is_safe.lower() == "true" else "false"
        logger.debug(f"Responder output: {response}, is_safe: {is_safe}")
        return dspy.Example(input=input_text, response=response, is_safe=is_safe).with_inputs('input', 'response')

    was_loaded = False
    if not force_regenerate and save_path is not None and os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                loaded_data = pickle.load(f)
            was_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load saved datasets: {e}")

        if was_loaded:
            try:
                assert isinstance(loaded_data, tuple) and len(loaded_data) == 2, "Invalid loaded datasets."
                assert all(isinstance(example, dspy.Example) for example in loaded_data[0]), "Invalid loaded trainset."
                assert all(isinstance(example, dspy.Example) for example in loaded_data[1]), "Invalid loaded valset."
                transformed_trainset, transformed_valset = loaded_data
            except:
                logger.warning("Invalid loaded datasets. Regenerating datasets for the evaluator module...")
                was_loaded = False

    if not was_loaded:
        input_trainset, input_valset = input_ds
        assert isinstance(input_trainset, list) and len(
            input_trainset) > 0, "Invalid input_trainset provided. Not a list or empty."
        assert isinstance(input_valset, list) and len(
            input_valset) > 0, "Invalid input_valset provided. Not a list or empty."
        assert all(isinstance(example, dspy.Example) for example in
                   input_trainset), "Invalid input_trainset provided. Not a list of dspy.Example."
        assert all(isinstance(example, dspy.Example) for example in
                   input_valset), "Invalid input_valset provided. Not a list of dspy.Example."

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            train_futures = [executor.submit(process_example, pipeline, example, logger) for example in input_trainset]
            val_futures = [executor.submit(process_example, pipeline, example, logger) for example in input_valset]

            transformed_trainset = [future.result() for future in concurrent.futures.as_completed(train_futures)]
            transformed_valset = [future.result() for future in concurrent.futures.as_completed(val_futures)]

        # Save the transformed datasets to disk
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump((transformed_trainset, transformed_valset), f)

    # Remove the 'input' field from the examples for the current implementation (not used as input to the Evaluator DSPy module)
    # transformed_trainset = [dspy.Example(response=example.response, is_safe=example.is_safe).with_inputs('response') for example in transformed_trainset]
    # transformed_valset = [dspy.Example(response=example.response, is_safe=example.is_safe).with_inputs('response') for example in transformed_valset]

    return transformed_trainset, transformed_valset


@hydra.main(config_path="configs", config_name="base_jail", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()
    # logger.info(f"Using configuration: {cfg}")

    with open(cfg.defense.unsafe_file, 'r') as f:
        unsafe_text = f.read()
    if unsafe_text is None:
       raise ValueError("The unsafe_text is None.")
    cfg.defense[cfg.defense.unsafe_subject] = unsafe_text

    pipeline = DSpyPipeline(cfg, logger)

    # Dynamically set the path for optimized
    cfg.model.dspy_optimized_file_orchestrator = os.path.basename(
        cfg.model.model_name) + f'-orchestrator' + cfg.model.dspy_optimized_file_postfix
    cfg.model.dspy_optimized_file_orchestrator = os.path.join(cfg.model.dspy_optimized_dir,
                                                              cfg.model.dspy_optimized_file_orchestrator)

    cfg.model.dspy_optimized_file_evaluator = os.path.basename(
        cfg.model.model_name) + f'-evaluator' + cfg.model.dspy_optimized_file_postfix
    cfg.model.dspy_optimized_file_evaluator = os.path.join(cfg.model.dspy_optimized_dir,
                                                           cfg.model.dspy_optimized_file_evaluator)

    cfg.model.evaluator_optimization_datasets_file = os.path.basename(
        cfg.model.model_name) + f'-evaluator' + cfg.model.dspy_dataset_file_postfix
    cfg.model.evaluator_optimization_datasets_file = os.path.join(cfg.model.dspy_optimized_common_dir,
                                                                  cfg.model.evaluator_optimization_datasets_file)

    # Prepare the training and validation data for the Orchestrator optimization
    dspy_trainset_orchestrator, dspy_valset_orchestrator = prepare_training_data(
        train_file=cfg.data.data.train_file,
        num_examples=50,
        train_ratio=0.2
    )
    pipeline.set_orchestrator_dspy_datasets(ds=(dspy_trainset_orchestrator, dspy_valset_orchestrator))

    # Prepare the training and validation data for the Orchestrator optimization
    dspy_trainset_evaluator, dspy_valset_evaluator = prepare_evaluator_training_ds(pipeline=pipeline,
                                                                                   input_ds=(dspy_trainset_orchestrator,
                                                                                             dspy_valset_orchestrator),
                                                                                   logger=logger,
                                                                                   max_workers=cfg.max_workers,
                                                                                   save_path=cfg.model.evaluator_optimization_datasets_file,
                                                                                   force_regenerate=False)
    pipeline.set_evaluator_dspy_datasets(ds=(dspy_trainset_evaluator, dspy_valset_evaluator))

    # Make directories for the optimization outputs
    os.makedirs(cfg.model.dspy_optimized_dir, exist_ok=True)
    cfg.enable_dspy_optimization = True
    initialization_time = str(time.time()).split('.')[0]
    if os.path.exists(cfg.model.dspy_optimized_file_orchestrator):
        os.rename(cfg.model.dspy_optimized_file_orchestrator,
                  cfg.model.dspy_optimized_file_orchestrator + '.bac' + initialization_time)
        logger.info(f"Copied existing optimization file for Orchestrator to: {cfg.model.dspy_optimized_file_orchestrator + '.bac' + initialization_time}")

    if os.path.exists(cfg.model.dspy_optimized_file_evaluator):
        os.rename(cfg.model.dspy_optimized_file_evaluator,
                  cfg.model.dspy_optimized_file_evaluator + '.bac' + initialization_time)
        logger.info(f"Copied existing optimization file for Evaluator to: {cfg.model.dspy_optimized_file_evaluator + '.bac' + initialization_time}")

    pipeline.run_optimize(force_retrain=True)
    logger.info("Optimization complete.")


if __name__ == "__main__":
    load_dotenv()
    main()

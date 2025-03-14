import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from sim.sim_jailbreak_llamaguard import JailbreakSimulator
import subprocess
from utils.logging import setup_logger

load_dotenv()


def run_evaluation(benchmark, input_path: str, output_dir: str, model: Optional[str] = None):
    subprocess_env = os.environ

    os.makedirs(output_dir, exist_ok=True)
    eval_script = os.path.join('evals', benchmark, f'eval_{benchmark}.py')
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    if benchmark == "false_refusal":
        subprocess_command = (
            f"python {eval_script} --input-path {input_path} --output-path {output_path} --model {model}")
    elif benchmark == "strong_reject":
        subprocess_command = (f"python {eval_script} --input-path {input_path} --output-path {output_path}")
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    print(f'Calling subprocess_command: {subprocess_command}')

    process = subprocess.Popen(subprocess_command, shell=True, env=subprocess_env)
    process.wait()
    return process.returncode


@hydra.main(config_path="configs", config_name="base_jail", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger = setup_logger()

    simulator = JailbreakSimulator(cfg, logger)

    benchmark = benchmark = cfg.data.data.name
    output_root = os.path.join('evals', benchmark, 'model_responses')
    max_workers = cfg.max_workers

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_basename = cfg.model.model_name.split('/')[-1]
    os.makedirs(output_root, exist_ok=True)
    output_file_basename = f'{benchmark}_responses_{timestamp}_{model_basename}'
    output_file = os.path.join(output_root, f'{output_file_basename}.csv')

    test_df = pd.read_csv(cfg.data.data.test_file)
    row_data = [{'benchmark': benchmark, 'row': row} for _, row in test_df.iterrows()]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulator.run, idx, row['row'], benchmark) for idx, row in enumerate(row_data)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts"):
            results.append(future.result())

    df = pd.DataFrame([result.__dict__ for result in results])
    df.to_csv(output_file, index=False, escapechar='\\', quoting=csv.QUOTE_ALL, doublequote=True)

    simulator.print_stats(results, len(test_df), output_file)

    # Call the run_evaluation function
    run_evaluation(benchmark, input_path=output_file, output_dir=os.path.join('evals', benchmark, 'model_evals'),
                   model=cfg.model.model_name)


if __name__ == "__main__":
    main()

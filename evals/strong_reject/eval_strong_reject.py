import os
import argparse
import pandas as pd
from strong_reject.evaluate import evaluate_dataset
from datasets import Dataset
from pathlib import Path
from dotenv import load_dotenv


def main(input_path: str, output_path: str, batch_size: int):
    # original_df = pd.read_csv("strongreject_small_jailbreak.csv")
    eval_df = pd.read_csv(input_path)

    dataset = Dataset.from_pandas(eval_df)
    print(dataset)
    eval_dataset = evaluate_dataset(dataset, ['strongreject_finetuned'], batch_size=batch_size)

    result_df = eval_dataset.to_pandas()

    os.makedirs(Path(output_path).parent, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(result_df)
    print(result_df["score"].mean())

    eval_df = pd.read_csv(output_path)
    print(eval_df["score"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate strong rejection responses.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size to be used for the evaluation.")
    args = parser.parse_args()
    # os.environ["TESTING"] = "1"
    load_dotenv()
    main(args.input_path, args.output_path, args.batch_size)

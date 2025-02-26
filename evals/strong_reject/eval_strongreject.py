from strong_reject.evaluate import evaluate_dataset
import pandas as pd
from datasets import Dataset
from pathlib import Path


eval_file = "jailbroken_responses_20250121_204407_Llama-3-8B-Instruct.csv"

df_org = pd.read_csv("strongreject_small_jailbreak.csv")
df = pd.read_csv(f"response/{eval_file}")


dataset = Dataset.from_pandas(df)

# breakpoint()
eval_dataset = evaluate_dataset(dataset, ["strongreject_finetuned"], batch_size=20)

result_df = eval_dataset.to_pandas()

result_df.to_csv(f"eval/eval_{eval_file}", index=False)

print(result_df["score"].mean())

# df = pd.read_csv(f"response/{eval_file}")
df = pd.read_csv(f"eval_{eval_file}")
print(df["score"].mean())

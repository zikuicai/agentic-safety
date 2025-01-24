from strong_reject.evaluate import evaluate_dataset
import pandas as pd
from datasets import Dataset
from pathlib import Path
# CUDA_VISIBLE_DEVICES=1

# root = Path("../../")
eval_file = "jailbroken_responses_20250121_204407_Llama-3.1-8B-Instruct.csv"
# eval_file = "jailbroken_responses_20250121_213432_Llama-3.1-8B-Instruct.csv"
# eval_file = "jailbroken_responses_20250121_212156_Llama-3-8B-Instruct-RR.csv"
# eval_file = "jailbroken_responses_20250121_214943_Llama-3-8B-Instruct-RR.csv"
# eval_file = "strong_reject_responses_20250122_222346_Llama-3-8B-Instruct-RR.csv"
# eval_file = "strong_reject_responses_20250122_233059_Llama-3-8B-Instruct-RR.csv"

# eval_file = "strong_reject_responses_20250123_105542_Llama-Guard-3-8B.csv"

# df_org = pd.read_csv("strongreject_small_jailbreak.csv")
# df = pd.read_csv(f"response/{eval_file}")

# # df['forbidden_prompt'] = df_org['forbidden_prompt']


# dataset = Dataset.from_pandas(df)
# # breakpoint()
# # for i in range(len(df)):
# #     if df['jailbroken_prompt'][i] != df_org['jailbroken_prompt'][i]:
# #         print(i), print(df['jailbroken_prompt'][i])
# #         break

# # breakpoint()
# eval_dataset = evaluate_dataset(dataset, ["strongreject_finetuned"], batch_size=20)

# result_df = eval_dataset.to_pandas()

# result_df.to_csv(f"eval/eval_{eval_file}", index=False)

# print(result_df["score"].mean())

# df = pd.read_csv(f"response/{eval_file}")
df = pd.read_csv(f"eval_{eval_file}")
print(df["score"].mean())

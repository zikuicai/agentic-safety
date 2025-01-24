import pandas as pd
from typing import List, Dict
import csv
from tqdm import tqdm
import os
from litellm import completion
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


openai_api_key = "token-abc123"
os.environ["OPENAI_API_KEY"] = openai_api_key
guard_model_id = "hosted_vllm/meta-llama/Llama-Guard-3-8B"
instruct_model_id = "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"

class SafetyChecker:
    def __init__(self, guard_api_base="http://localhost:8001/v1", instruct_api_base="http://localhost:8000/v1"):
        self.guard_api_base = guard_api_base
        self.instruct_api_base = instruct_api_base
        # Track cumulative safety statistics
        self.total_checks = 0
        self.safe_first_checks = 0
        self.safe_second_checks = 0

    def check_safety(self, chat: List[Dict[str, str]]) -> bool:
        try:
            response = completion(
                model=guard_model_id,
                messages=chat,
                api_base=self.guard_api_base
            )
            result = response.choices[0].message.content
            return "safe" in result.strip().lower() and "unsafe" not in result.strip().lower()
        except Exception as e:
            print(f"Safety check error: {e}")
            return False

    def generate_refusal_message(self, stage: str) -> str:
        refusal_messages = {
            "first_check": "I'm sorry, but I can't assist with that.",
            "second_check": "I'm sorry, but I can't assist with that."
        }
        return refusal_messages.get(stage, "Request cannot be processed due to safety concerns.")

    def get_instruction_response(self, prompt: str) -> str:
        try:
            response = completion(
                model=instruct_model_id,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.instruct_api_base
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Instruction model error: {e}")
            return ""

    def process_single_prompt(self, row_data):
        self.total_checks += 1
        
        category = row_data['category']
        row = row_data['row']
        
        if category == "strong_reject":
            prompt = row['jailbroken_prompt']
            forbidden_prompt = row['forbidden_prompt']
        else:
            prompt = row['prompt']
            forbidden_prompt = None

        first_check = self.check_safety([{"role": "user", "content": prompt}])
        if first_check:
            self.safe_first_checks += 1
            response = self.get_instruction_response(prompt)
            second_check = self.check_safety([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
            if second_check:
                self.safe_second_checks += 1
            else:
                response = self.generate_refusal_message("second_check")
        else:
            response = self.generate_refusal_message("first_check")
            second_check = False

        # Calculate current safety ratios
        first_check_ratio = self.safe_first_checks / self.total_checks
        second_check_ratio = self.safe_second_checks / self.total_checks

        print(f"first_check_ratio: {first_check_ratio}")
        print(f"second_check_ratio: {second_check_ratio}")

        result = {
            "prompt": prompt,
            "first_check_safe": first_check,
            "response": response,
            "second_check_safe": second_check,
            "first_check_ratio": first_check_ratio,
            "second_check_ratio": second_check_ratio
        }
        
        if forbidden_prompt is not None:
            result["forbidden_prompt"] = forbidden_prompt
            
        return result

    def process_csv(self, input_file: str, output_file: str = None, max_workers: int = 4):
        category = input_file.split("/")[-2]
        
        if output_file is None:
            output_root = f"evals/{category}/response"
            os.makedirs(output_root, exist_ok=True)
            output_file = os.path.join(output_root, "safety_results.csv")
        
        test_df = pd.read_csv(input_file)
        row_data = [{'category': category, 'row': row} for _, row in test_df.iterrows()]
        
        # Reset safety statistics before processing new file
        self.total_checks = 0
        self.safe_first_checks = 0
        self.safe_second_checks = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_single_prompt, row_data),
                total=len(row_data),
                desc="Processing prompts"
            ))
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, 
            index=False,
            escapechar='\\',
            quoting=csv.QUOTE_ALL,
            doublequote=True)
        
        self._print_stats(results, len(test_df), output_file)

    def _print_stats(self, results, total_prompts, output_file):
        first_round_safe = sum(1 for r in results if r["first_check_safe"])
        second_round_safe = sum(1 for r in results if r["second_check_safe"])
        
        print(f"Results saved to: {output_file}")
        print(f"Total prompts: {total_prompts}")
        print(f"First round safe: {first_round_safe} ({first_round_safe/total_prompts*100:.2f}%)")
        print(f"Second round safe: {second_round_safe} ({second_round_safe/total_prompts*100:.2f}%)")
        print(f"Cumulative first check safety ratio: {self.safe_first_checks/self.total_checks:.4f}")
        print(f"Cumulative second check safety ratio: {self.safe_second_checks/self.total_checks:.4f}")


# Example usage
if __name__ == "__main__":
    checker = SafetyChecker()
    
    # input_file = "evals/strong_reject/strongreject_small_jailbreak.csv"
    input_file = "evals/false_refusal/PHTest_harmless.csv"
    
    category = input_file.split("/")[-2]
    output_root = f"evals/{category}/response"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_basename = "Llama-Guard-3-8B"
    output_file = f'{output_root}/{category}_responses_{timestamp}_{model_basename}.csv'

    checker.process_csv(input_file, output_file)
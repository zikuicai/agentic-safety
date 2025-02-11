import pandas as pd
import numpy as np
from random import sample, seed
import os

def save_methods_info(methods, scores, output_path):
    """Save selected methods and their scores to a text file."""
    methods_file = f"{os.path.splitext(output_path)[0]}_methods.txt"
    with open(methods_file, 'w') as f:
        f.write("Selected Methods and Scores:\n")
        f.write("-" * 30 + "\n")
        for method in methods:
            f.write(f"{method}: {scores[method]:.6f}\n")
    return methods_file

def get_methods(methods_dict, num_methods, random_seed, previous_methods=None):
    """Get randomly selected methods, optionally including previous selections."""
    seed(random_seed)
    
    if previous_methods:
        remaining_methods = [m for m in methods_dict.keys() if m not in previous_methods]
        additional_methods = sample(remaining_methods, num_methods - len(previous_methods))
        return previous_methods + additional_methods
    else:
        return sample(list(methods_dict.keys()), num_methods)

def create_train_eval_split(df, method, train_samples, eval_samples, random_seed):
    """Split method samples into train and evaluation sets."""
    method_prompts = df[df['jailbreak'] == method]
    if len(method_prompts) < (train_samples + eval_samples):
        print(f"Warning: Method {method} has fewer than {train_samples + eval_samples} prompts")
        # Split available samples proportionally
        total_available = len(method_prompts)
        train_prop = train_samples / (train_samples + eval_samples)
        actual_train = int(total_available * train_prop)
        actual_eval = total_available - actual_train
        shuffled = method_prompts.sample(frac=1, random_state=random_seed)
        return shuffled.iloc[:actual_train], shuffled.iloc[actual_train:]
    else:
        shuffled = method_prompts.sample(n=train_samples + eval_samples, random_state=random_seed)
        return shuffled.iloc[:train_samples], shuffled.iloc[train_samples:]

def create_datasets(input_path, phtest_path, output_dir, methods_dict, random_seed, samples_per_method=5, eval_samples_per_method=10):
    """
    Create training datasets (n=5,10,15), ID evaluation dataset (unseen samples from training methods),
    OOD evaluation dataset (samples from unseen methods), and benign evaluation dataset.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(random_seed)
        
        # Read the CSV files
        df = pd.read_csv(input_path)
        phtest_df = pd.read_csv(phtest_path)
        
        # Reserve benign samples for evaluation
        total_benign_needed = len(methods_dict) * eval_samples_per_method
        eval_benign = phtest_df.sample(n=total_benign_needed, random_state=random_seed)
        train_benign = phtest_df.drop(eval_benign.index)
        
        # Dictionary to store evaluation samples for each method
        eval_samples_dict = {}
        
        # Create training datasets for n=5,10,15
        previous_methods = None
        all_train_data = {}  # Store all training splits for different n
        
        for n in [5, 10, 15]:
            print(f"\nCreating training dataset for n={n}...")
            
            # Select methods
            selected_methods = get_methods(methods_dict, n, random_seed, previous_methods)
            previous_methods = selected_methods
            
            # Prepare output paths
            train_output = os.path.join(output_dir, f"train_n{n}_seed{random_seed}.csv")
            methods_info = save_methods_info(selected_methods, methods_dict, train_output)
            
            # Sample and save training data
            train_rows = []
            for method in selected_methods:
                train_data, eval_data = create_train_eval_split(
                    df, method, samples_per_method, eval_samples_per_method, random_seed
                )
                train_rows.append(train_data)
                eval_samples_dict[method] = eval_data
            
            # Combine harmful training data
            harmful_df = pd.concat(train_rows, ignore_index=True)
            
            # Sample benign prompts for training
            benign_sample = train_benign.sample(n=len(harmful_df), random_state=random_seed)
            
            # Create and save training dataset
            train_df = pd.DataFrame({
                'category': ['harmful'] * len(harmful_df) + ['benign'] * len(benign_sample),
                'prompt': list(harmful_df['jailbroken_prompt']) + list(benign_sample['prompt']),
                'method': list(harmful_df['jailbreak']) + ['benign'] * len(benign_sample)
            })
            train_df.to_csv(train_output, index=False)
            print(f"Created training dataset: {train_output}")
            
            # Store training methods for each n
            all_train_data[n] = {
                'methods': selected_methods,
                'eval_samples': eval_samples_dict.copy()
            }
        
        # Create evaluation datasets for each training set size
        for n in [5, 10, 15]:
            train_info = all_train_data[n]
            trained_methods = train_info['methods']
            
            # ID evaluation (unseen samples from training methods)
            print(f"\nCreating ID evaluation dataset for n={n}...")
            id_eval_rows = []
            for method in trained_methods:
                if method in train_info['eval_samples']:
                    id_eval_rows.append(train_info['eval_samples'][method])
            
            if id_eval_rows:
                id_harmful_df = pd.concat(id_eval_rows, ignore_index=True)
                id_eval_df = pd.DataFrame({
                    'category': ['harmful'] * len(id_harmful_df),
                    'prompt': list(id_harmful_df['jailbroken_prompt']),
                    'forbidden_prompt': list(id_harmful_df['forbidden_prompt']),
                    'method': list(id_harmful_df['jailbreak'])
                })
                
                id_eval_path = os.path.join(output_dir, f"eval_id_n{n}_seed{random_seed}.csv")
                id_eval_df.to_csv(id_eval_path, index=False)
                print(f"Created ID evaluation dataset: {id_eval_path}")
            
            # OOD evaluation (samples from unseen methods)
            print(f"\nCreating OOD evaluation dataset for n={n}...")
            ood_rows = []
            unseen_methods = [m for m in methods_dict.keys() if m not in trained_methods]
            
            for method in unseen_methods:
                method_prompts = df[df['jailbreak'] == method]
                if len(method_prompts) >= eval_samples_per_method:
                    ood_rows.append(method_prompts.sample(n=eval_samples_per_method, random_state=random_seed))
                else:
                    print(f"Warning: Method {method} has fewer than {eval_samples_per_method} prompts")
                    ood_rows.append(method_prompts)
            
            if ood_rows:
                ood_harmful_df = pd.concat(ood_rows, ignore_index=True)
                ood_eval_df = pd.DataFrame({
                    'category': ['harmful'] * len(ood_harmful_df),
                    'prompt': list(ood_harmful_df['jailbroken_prompt']),
                    'forbidden_prompt': list(ood_harmful_df['forbidden_prompt']),
                    'method': list(ood_harmful_df['jailbreak'])
                })
                
                ood_eval_path = os.path.join(output_dir, f"eval_ood_n{n}_seed{random_seed}.csv")
                ood_eval_df.to_csv(ood_eval_path, index=False)
                print(f"Created OOD evaluation dataset: {ood_eval_path}")
        
        # Create benign evaluation dataset
        print("\nCreating benign evaluation dataset...")
        benign_eval_df = pd.DataFrame({
            'category': ['benign'] * len(eval_benign),
            'prompt': list(eval_benign['prompt']),
            'forbidden_prompt': [''] * len(eval_benign),  # Empty string for benign prompts
            'method': ['benign'] * len(eval_benign)
        })
        
        benign_eval_path = os.path.join(output_dir, f"eval_benign_seed{random_seed}.csv")
        benign_eval_df.to_csv(benign_eval_path, index=False)
        print(f"Created benign evaluation dataset: {benign_eval_path}")
        
        # Print dataset statistics and method distributions
        print("\nDataset Statistics:")
        for n in [5, 10, 15]:
            print(f"\nFor n={n}:")
            print(f"Training methods: {', '.join(all_train_data[n]['methods'])}")
            print(f"ID eval methods: {', '.join(all_train_data[n]['methods'])}")
            unseen_methods = [m for m in methods_dict.keys() if m not in all_train_data[n]['methods']]
            if unseen_methods:
                print(f"OOD eval methods: {', '.join(unseen_methods)}")
            
            train_path = os.path.join(output_dir, f"train_n{n}_seed{random_seed}.csv")
            id_eval_path = os.path.join(output_dir, f"eval_id_n{n}_seed{random_seed}.csv")
            ood_eval_path = os.path.join(output_dir, f"eval_ood_n{n}_seed{random_seed}.csv")
            
            train_df = pd.read_csv(train_path)
            id_eval_df = pd.read_csv(id_eval_path) if os.path.exists(id_eval_path) else pd.DataFrame()
            ood_eval_df = pd.read_csv(ood_eval_path) if os.path.exists(ood_eval_path) else pd.DataFrame()
            
            print(f"Training samples: {len(train_df)}")
            print(f"ID eval samples: {len(id_eval_df)}")
            print(f"OOD eval samples: {len(ood_eval_df)}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e



# Define the methods and their scores
methods = {
    'evil_confidant': 0.548682,
    'dev_mode_v2': 0.538862,
    'pair': 0.466778,
    'dev_mode_with_rant': 0.402161,
    'aim': 0.400528,
    'pap_misrepresentation': 0.324022,
    'pap_authority_endorsement': 0.317738,
    'pap_logical_appeal': 0.299832,
    'pap_expert_endorsement': 0.284874,
    'refusal_suppression': 0.233648,
    'pap_evidence_based_persuasion': 0.233004,
    'distractors_negated': 0.228428,
    'disemvowel': 0.181410,
    'combination_3': 0.157014,
    'translation_guarani': 0.156256
}

# Usage example
if __name__ == "__main__":
    create_datasets(
        input_path="../strong_reject/strongreject_small_jailbreak.csv",
        phtest_path="PHTest_harmless.csv",
        # output_path="sampled_dataset_n5.csv",
        output_dir="dataset_splits",
        methods_dict=methods,
        samples_per_method=5,
        eval_samples_per_method=20,
        random_seed=42,
    )
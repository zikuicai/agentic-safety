import pandas as pd

def transform_csv_files(phtest_path, strongreject_path, output_path, num_entries=2000):
    """
    Transform and combine CSV files according to specified requirements.
    
    Args:
        phtest_path (str): Path to the PHTest CSV file
        strongreject_path (str): Path to the strongreject CSV file
        output_path (str): Path for the output CSV file
        num_entries (int): Number of entries to take from each file
    """
    try:
        # Read the CSV files
        harmless_df = pd.read_csv(phtest_path)
        harmful_df = pd.read_csv(strongreject_path)
        
        # Take first num_entries from each
        harmless_df = harmless_df.head(num_entries)
        harmful_df = harmful_df.head(num_entries)
        
        # Create new dataframe with required structure
        harmful_transformed = pd.DataFrame({
            'category': ['harmful'] * num_entries,
            'prompt': harmful_df['jailbroken_prompt']
        })
        
        # Concatenate the dataframes
        result_df = pd.concat([harmless_df, harmful_transformed], ignore_index=True)
        
        # Save to new CSV file
        result_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path}")
        print(f"Total rows in output: {len(result_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage example
if __name__ == "__main__":
    transform_csv_files(
        phtest_path="PHTest_harmless.csv",
        strongreject_path="../strong_reject/strongreject_small_jailbreak.csv",
        output_path="combined_dataset.csv"
    )
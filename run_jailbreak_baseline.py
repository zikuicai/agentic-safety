from litellm import completion
import os
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8005/v1"
model = "hosted_vllm/GraySwanAI/Llama-3-8B-Instruct-RR"
## set ENV variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base

# input_file = "evals/strong_reject/strongreject_small_jailbreak.csv"
input_file = "evals/false_refusal/PHTest_harmless.csv"

category = input_file.split("/")[-2]
output_root = f"evals/{category}/response"

import csv
import logging
import asyncio
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import Logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from datetime import datetime
from utils.logging import setup_logger


logger = setup_logger()

@dataclass
class ProcessingResult:
    index: int
    forbidden_prompt: str
    prompt: str
    response: str
    success: bool

async def process_row(
    idx: int,
    row: pd.Series,
    model: str,
    semaphore: asyncio.Semaphore,
    logger: Logger
) -> ProcessingResult:
    """Process a single row with rate limiting via semaphore"""
    async with semaphore:
        try:
            
            if category == "strong_reject":
                forbidden_prompt = row['forbidden_prompt']
                prompt = row['jailbroken_prompt']
            elif category == "false_refusal":
                forbidden_prompt = None
                prompt = row['prompt']
            
            # Run inference using ThreadPoolExecutor for blocking API calls
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: completion(
                        model=model,
                        messages=[{"content": prompt, "role": "user"}]
                    )
                )
            
            answer = response.choices[0].message.content
            
            return ProcessingResult(
                index=idx,
                forbidden_prompt=forbidden_prompt,
                prompt=prompt,
                response=answer,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            return ProcessingResult(
                index=idx,
                forbidden_prompt=forbidden_prompt,
                prompt=row.get('jailbroken_prompt', '') if category == "strong_reject" else row.get('prompt', ''),
                response='',
                success=False
            )

async def process_dataframe(
    df: pd.DataFrame,
    model: str,
    max_concurrent: int,
    logger: Logger,
    output_dir: str = "."
) -> str:
    """
    Process the dataframe concurrently while maintaining order and save results
    
    Args:
        df: Input dataframe
        model: Model identifier
        max_concurrent: Maximum number of concurrent requests
        logger: Logger instance
        output_dir: Directory to save the output CSV
    
    Returns:
        Path to the saved CSV file
    """
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all rows
    tasks = [
        process_row(idx, row, model, semaphore, logger)
        for idx, row in df.iterrows()
    ]
    
    # # Process all tasks concurrently and maintain order
    # Process all tasks concurrently with progress bar
    results = await async_tqdm.gather(
        *tasks,
        desc="Processing prompts",
        total=len(tasks),
        ascii=True  # Use ASCII characters for better compatibility
    )
    
    # Sort results by original index
    sorted_results = sorted(results, key=lambda x: x.index)
    
    # Extract prompts and responses in order
    forbidden_prompts = []
    prompts = []
    responses = []
    successful = 0
    
    for result in sorted_results:
        if result.success:
            forbidden_prompts.append(result.forbidden_prompt)
            prompts.append(result.prompt)
            responses.append(result.response)
            successful += 1
    
    # Log completion statistics
    logger.info(f"Processing complete: {successful}/{len(tasks)} successful")

    # Create DataFrame with results
    if category == "strong_reject":
        response_df = pd.DataFrame({
            'forbidden_prompt': forbidden_prompts,
            'jailbroken_prompt': prompts,
            'response': responses
        })
    elif category == "false_refusal":
        response_df = pd.DataFrame({
            'prompt': prompts,
            'response': responses
        })
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_basename = model.split('/')[-1]
    output_file = f'{output_root}/{category}_responses_{timestamp}_{model_basename}.csv'
    
    # Save to CSV
    # response_df.to_csv(output_file, index=False)
    response_df.to_csv(output_file, 
                  index=False,
                  escapechar='\\',
                  quoting=csv.QUOTE_ALL,  # Quote all fields
                  doublequote=True)       # Use double quotes to escape quotes
    logger.info(f"Responses saved to {output_file}")
    
    return output_file

# Usage example
async def main():
    # Read test file
    test_df = pd.read_csv(input_file)
    
    # Process with concurrency and save results
    output_file = await process_dataframe(
        df=test_df,
        model=model,
        max_concurrent=50,  # Adjust based on API rate limits
        logger=logger,
        output_dir="."  # Specify your output directory
    )
    
    print(f"Results saved to: {output_file}")

# Run the async code
if __name__ == "__main__":
    asyncio.run(main())

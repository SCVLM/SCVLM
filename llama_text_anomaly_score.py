import transformers
import torch
import pandas as pd
import os
import time
from requests.exceptions import SSLError

# File paths
input_csv = ''
output_csv = 'score.csv'
last_img = None
allowed_behaviors_csv = 'rule_refined.csv'  # File containing allowed behaviors

# Initialize LLAMA model and pipeline
model_id = "LLAMA3.1"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Load allowed behaviors from the CSV
with open(allowed_behaviors_csv, 'r') as f:
    allowed_behaviors = f.readline().strip()
    print(allowed_behaviors)

# Define the function to evaluate description similarity
def rate_score(answer_org):
    prompt = f"""Allowed Behaviors:  
{allowed_behaviors}  

This is a guideline that includes \"Allowed Behaviors\". Please use it to rate the following description from 0-10, where 0 indicates the scene contains only \"Allowed Behaviors\" and 10 indicates containing at least one \"Prohibited Behavior\". Note that any human activity not listed explicitly in \"Allowed Behaviors\" is a \"Prohibited Behavior\".\n
**Description**: {answer_org}\n
Please provide your rating and a brief explanation in the following format:  \nScore: X  \nExplanation: Y"""

    messages = [
        {"role": "system", "content": "You are a video surveillance analyst, responsible for understanding and analyzing video content based on its descriptions."},
        {"role": "user", "content": prompt}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        do_sample=False
    )

    # Extract generated text
    llama_response = outputs[0]['generated_text'][2]['content']

    # Extract score and explanation from response
    score = None
    explanation = None
    for line in llama_response.split('\n'):
        if line.startswith("Score:"):
            score = int(line.split("Score:")[1].strip())
        if line.startswith("Explanation:"):
            explanation = line.split("Explanation:")[1].strip()

    return score, explanation

# Load input CSV
df = pd.read_csv(input_csv)

# Initialize results list
results = []

# Check if output CSV exists and load existing results
if os.path.exists(output_csv):
    existing_results_df = pd.read_csv(output_csv)
    processed_imgs = set(existing_results_df['image_path'].tolist())
else:
    existing_results_df = pd.DataFrame()
    processed_imgs = set()

# Start processing based on last_img
start_processing = last_img is None

for index, row in df.iterrows():
    image_path, memloss, label, answer_org, answer_mem = row
    #image_path, memloss, label, answer_org = row

    # If last_img is specified, skip until that image is found
    if not start_processing:
        if image_path == last_img:
            start_processing = True
        continue

    # Skip images already processed in the output CSV
    if image_path in processed_imgs:
        continue

    # Call LLAMA model to compare and get score and explanation
    score, explanation = rate_score(answer_org)
    print(score)
    print(explanation)

    print(f"Processing {image_path}...")

    # Append result to the list
    results.append([image_path, memloss, label, score, explanation])

    # Update last_img
    last_img = image_path

    # Save incrementally to avoid data loss
    new_results_df = pd.DataFrame(results, columns=['image_path', 'memloss', 'label', 'score', 'explanation'])
    new_results_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
    results.clear()  # Clear results after saving to avoid duplicates

print(f"Evaluation complete. Results saved to {output_csv}.")
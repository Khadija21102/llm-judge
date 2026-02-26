## Distribution analysis
# This script reads a JSONL dataset, rounds the scores to the nearest integer, clamps them to be between 1 and 5, and then samples an equal number of examples for each score to create a balanced dataset. Finally, it saves the new balanced dataset to a new JSONL file.

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_json("dataset_ref_based_scores_train_new_v2.jsonl", lines=True)

# Compute means
data = df['output'].apply(lambda x: np.round(x))

# Clamp values < 1 to 1, > 5 to 5
data_clamped = data.clip(lower=1, upper=5)

# Add clamped values back to the dataframe
df['output'] = data_clamped
#df['output'] = data

# Count occurrences of each score
value_counts = df['output'].value_counts().sort_index()

print("Counts per score:", value_counts.to_dict())

# Find the maximum possible uniform sample size
max_per_class = value_counts.min()
print("Max samples per class:", max_per_class)

# Sample equally from each class
balanced_df_list = []
for value in value_counts.index:
    subset = df[df['output'] == value]
    sampled = subset.sample(max_per_class, random_state=42)
    balanced_df_list.append(sampled)

balanced_df = pd.concat(balanced_df_list)

# Save new balanced dataset
balanced_df.to_json("dataset_ref_based_scores_train_new_balanced_v2.jsonl", 
                    orient="records", 
                    lines=True)

print("Balanced dataset saved with", len(balanced_df), "samples.")

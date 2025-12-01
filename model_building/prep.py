
# tourism_project/model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from huggingface_hub import HfApi

# Initialize HF API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Correct dataset repo (hyphen!)
HF_DATASET_ID = "shishirtiwari/tourism-project"

# Load dataset using HuggingFace datasets library (correct way)
print("Downloading dataset from Hugging Face...")
dataset = load_dataset(HF_DATASET_ID)

# If your dataset is a CSV, datasets loads it under the 'train' split
df = dataset["train"].to_pandas()
print("Dataset loaded successfully.")

# Drop ID col
df.drop(columns=['CustomerID'], inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()

encode_cols = [
    "TypeofContact", "CityTier", "Occupation",
    "Gender", "ProductPitched", "MaritalStatus",
    "Designation"
]

for col in encode_cols:
    df[col] = encoder.fit_transform(df[col])

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

# Split dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save outputs
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload splits to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=HF_DATASET_ID,   # FIXED
        repo_type="dataset",
    )

print("Dataset splits uploaded successfully.")

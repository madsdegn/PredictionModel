# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from src.train import train_model
from src.models import MODEL_REGISTRY

# Initialize list to store metrics for each model
results = []

# Loop through all models defined in the registry
for name in MODEL_REGISTRY.keys():
    # Print which model is currently being trained
    print(f"\nTraining {name}...")

    # Train the model and collect evaluation metrics
    metrics = train_model(model_name=name)

    # Append metrics to results list, including the model name
    results.append({"Model": name, **metrics})

# Convert results list into a pandas DataFrame for easier handling
df = pd.DataFrame(results)

# Define the dataset splits we want to display (train, validation, test)
splits = ["Train", "Validation", "Test"]

# Loop through each split and print corresponding results
for split in splits:
    # Select columns that belong to this split (e.g., Train_Accuracy, Train_F1, etc.)
    split_cols = [c for c in df.columns if c.startswith(split)]

    # Print header for the split
    print(f"\n{split} Results:")

    # Print metrics for all models in this split
    print(df[["Model"] + split_cols])
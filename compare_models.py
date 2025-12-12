# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from src.train import train_model
from src.models import MODEL_REGISTRY

results = []  # Initialize list to store metrics for each model

# Loop through all models defined in the registry
for name in MODEL_REGISTRY.keys():
    print(f"\nTraining {name}...")  # Print which model is being trained
    metrics = train_model(model_name=name)  # Train the model and collect metrics
    results.append({"Model": name, **metrics})  # Append metrics with model name to results list

df = pd.DataFrame(results)  # Convert results list into a pandas DataFrame

# Define the dataset splits we want to display
splits = ["Train", "Validation", "Test"]

# Loop through each split and print corresponding results
for split in splits:
    split_cols = [c for c in df.columns if c.startswith(split)]  # Select columns for this split
    print(f"\n{split} Results:")  # Print header for the split
    print(df[["Model"] + split_cols])  # Print metrics for all models in this split
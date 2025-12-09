# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from src.train import train_model
from src.models import MODEL_REGISTRY

results = []

for name in MODEL_REGISTRY.keys():
    print(f"\nTraining {name}...")
    metrics = train_model(model_name=name)  # modify train_model to return metrics
    results.append({"Model": name, **metrics})

df = pd.DataFrame(results)
print("\nModel Comparison:")
print(df)
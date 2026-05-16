import torch
from torch_geometric.data import Data
import os

# assume your modified data.py is in the same directory or accessible via import
from data import get_credit, get_income

# Adjust this path to where your dataset folder resides
root = "~/data/seed_gnn_data/dataset"

# ---- Test Credit ----
print("=== Testing CREDIT dataset ===")
credit_data, credit_nf, credit_nc = get_credit(root)
print(f"Num features: {credit_nf}, Num classes: {credit_nc}")
print(f"Num nodes: {credit_data.num_nodes}, Num edges: {credit_data.edge_index.size(1)}")
print(f"Feature matrix shape: {credit_data.x.shape}")
print(f"Train/Val/Test split: {credit_data.train_mask.sum().item()}/{credit_data.val_mask.sum().item()}/{credit_data.test_mask.sum().item()}")
print(f"Label distribution: {torch.bincount(credit_data.y)}")
print(f"Sensitive attr mean/std: {credit_data.sens.mean().item():.3f} / {credit_data.sens.std().item():.3f}")
print()

# ---- Test Income ----
print("=== Testing INCOME dataset ===")
income_data, income_nf, income_nc = get_income(root)
print(f"Num features: {income_nf}, Num classes: {income_nc}")
print(f"Num nodes: {income_data.num_nodes}, Num edges: {income_data.edge_index.size(1)}")
print(f"Feature matrix shape: {income_data.x.shape}")
print(f"Train/Val/Test split: {income_data.train_mask.sum().item()}/{income_data.val_mask.sum().item()}/{income_data.test_mask.sum().item()}")
print(f"Label distribution: {torch.bincount(income_data.y)}")
print(f"Sensitive attr mean/std: {income_data.sens.mean().item():.3f} / {income_data.sens.std().item():.3f}")
print()

# Quick sanity check
assert credit_data.x.size(0) == credit_data.y.size(0), "Credit: x and y size mismatch!"
assert income_data.x.size(0) == income_data.y.size(0), "Income: x and y size mismatch!"
print("✅ Both datasets loaded successfully and passed basic sanity checks.")

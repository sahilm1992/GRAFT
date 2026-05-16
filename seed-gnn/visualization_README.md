# Feature Corruption Visualization

This document explains how to use the enhanced `visualize_validation` function that creates comprehensive visualizations for different feature corruption scenarios.

## Overview

The new visualization function creates three main types of plots, each with subplots for different feature corruption scenarios:

1. **ROC/AUC Curves** - Shows classification performance for each class under different corruption scenarios
2. **Misclassification Rate vs Node Degree** - Shows how misclassification varies with node degree
3. **True Class Probability vs Node Degree** - Shows confidence in predictions vs node degree

## Configuration

To specify which features should be corrupted, add a `feature_corruption` section to your configuration:

```json
{
    "eval_params": {
        "dataset": "cora",
        "num_targets": 50
    },
    "feature_corruption": {
        "features": [0, 1, 2, 3, 4, 5, 6, 7]
    },
    "management": {
        "output_folder_dir": "path/to/output"
    }
}
```

### Feature Corruption Options

- **Specific features**: `"features": [0, 1, 2]` - Only corrupt features 0, 1, and 2
- **All features**: `"features": "all"` - Corrupt all available features
- **No corruption**: Omit the section entirely - Only show original data

## Feature Corruption Types

The function automatically detects feature types and applies appropriate corruption:

- **Boolean features**: Values are flipped (0→1, 1→0)
- **Discrete/Continuous features**: Values are shuffled randomly

## Output

All plots are saved in a `visualization_plots` subdirectory within your specified output folder:

```
output_folder/
├── visualization_plots/
│   ├── roc_auc_curves.png
│   ├── misclassification_rate_vs_degree.png
│   └── true_class_probability_vs_degree.png
├── val_results.txt
└── other_outputs...
```

## Generated Scenarios

The function automatically generates all combinations of feature corruption:

- **Original data**: No corruption
- **Single feature corruption**: Each feature corrupted individually
- **Multiple feature combinations**: All possible combinations of 2, 3, 4, etc. features

For example, with features [0, 1, 2], it will create:
- Original data
- Corrupt feature 0 only
- Corrupt feature 1 only  
- Corrupt feature 2 only
- Corrupt features 0+1
- Corrupt features 0+2
- Corrupt features 1+2
- Corrupt features 0+1+2

## Usage

The function is called automatically during validation. Simply ensure your configuration includes the `feature_corruption` section if you want feature corruption analysis.

## Dependencies

The visualization requires:
- `sklearn.metrics` for ROC/AUC calculations
- `matplotlib` for plotting
- `torch` for model predictions
- `numpy` for data manipulation


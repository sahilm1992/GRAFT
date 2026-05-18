# Editing Pipelines Framework

A flexible framework for implementing and testing different GNN editing methods. This framework provides a common interface for loading models, running validation, performing edits, and visualizing results.

## Features

- **Modular Design**: Easy to extend with new editing methods
- **Shared Utilities**: Common functions for model loading, validation, testing, and visualization
- **Base Editor Class**: Abstract base class that handles common functionality
- **Multiple Methods**: Implementations of EGNN and SEED-GNN editing methods
- **Comprehensive Testing**: Built-in test suite for framework validation

## Architecture

### BaseEditor Class

The `BaseEditor` class provides common functionality for all editing methods:

- Model and data loading
- Before-edit evaluation
- Fine-tuning for GNN+MLP architectures
- Edit target selection
- Result processing and visualization
- Configuration management

### Editor Implementations

#### EGNNEditor
Implements the basic EGNN (Edit GNN) method where each node is edited independently using gradient-based optimization.

#### SEEDGNEditor
Implements the SEED-GNN method which uses mixup training with neighborhood nodes to improve editing performance.

## Installation

The framework requires the seed-gnn codebase to be available. Make sure the seed-gnn directory is properly set up with all dependencies.

## Usage

### Basic Usage

```python
from editing_pipelines import EGNNEditor, SEEDGNEditor

# Load configuration
config = {
    'management': {
        'dataset_dir': 'path/to/data',
        'pretrain_output_dir': 'path/to/pretrained',
        'output_folder_dir': 'path/to/output'
    },
    'eval_params': {
        'dataset': 'cora',
        'num_targets': 10
    },
    'pipeline_params': {
        'model_name': 'GCN',
        'optim': 'adam',
        'edit_lr': 0.01,
        'max_num_edit_steps': 20,
        'architecture': {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5
        }
    }
}

# Create editor
editor = EGNNEditor(config)

# Run editing experiment
raw_results, processed_results = editor.run_editing_experiment()
```

### Using the Factory Function

```python
from editing_pipelines import create_editor

# Create editor using factory
editor = create_editor('egnn', config)
# or
editor = create_editor('seed_gnn', config)
```

### Custom Editing Methods

To implement a new editing method, subclass `BaseEditor` and implement the `edit_model` method:

```python
from editing_pipelines import BaseEditor

class MyCustomEditor(BaseEditor):
    def edit_model(self, node_idx_2flip, flipped_label, max_num_step):
        # Implement your editing strategy here
        raw_results = []
        
        for idx, label in zip(node_idx_2flip, flipped_label):
            # Your editing logic
            result = self.perform_edit(idx, label, max_num_step)
            raw_results.append(result)
        
        return raw_results
```

## Configuration

The framework uses a configuration dictionary with the following structure:

```python
config = {
    'management': {
        'dataset_dir': str,           # Path to datasets
        'pretrain_output_dir': str,   # Path to pretrained models
        'output_folder_dir': str      # Path for output files
    },
    'eval_params': {
        'dataset': str,              # Dataset name (cora, citeseer, etc.)
        'num_targets': int,          # Number of nodes to edit
        'subgroup_features': list     # Features for subgroup analysis
    },
    'pipeline_params': {
        'model_name': str,           # Model type (GCN, GAT, etc.)
        'optim': str,                # Optimizer (adam, rmsprop)
        'edit_lr': float,            # Learning rate for editing
        'max_num_edit_steps': int,   # Maximum editing steps
        'architecture': dict,        # Model architecture parameters
        'alpha': float,              # SEED-GNN mixup ratio
        'beta': int                  # SEED-GNN number of mixup samples
    },
    'corruption': {
        'features': list             # Features to corrupt for analysis
    }
}
```

## Testing

Run the test suite to verify the framework:

```bash
cd /home/model_editing/GRAFT/editing_pipelines
python test_framework.py
```

The test suite includes:
- Editor creation tests
- Model loading tests
- Evaluation tests
- Edit target selection tests
- Simple editing experiment

## File Structure

```
editing_pipelines/
├── __init__.py              # Package initialization and exports
├── base_editor.py           # Base editor class
├── egnn_editor.py          # EGNN editor implementation
├── seed_gnn_editor.py      # SEED-GNN editor implementation
├── utils.py                 # Shared utility functions
├── test_framework.py       # Test suite
└── README.md               # This file
```

## Dependencies

The framework depends on:
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- The seed-gnn codebase

## Contributing

To add a new editing method:

1. Create a new file `your_method_editor.py`
2. Subclass `BaseEditor`
3. Implement the `edit_model` method
4. Add any method-specific parameters and methods
5. Update `__init__.py` to export your new editor
6. Add tests to `test_framework.py`

## License

This framework is part of the GNN editing exploration project.

"""
Editing Pipelines Package

This package provides a framework for implementing and testing different GNN editing methods.
It includes:

- BaseEditor: Abstract base class for all editing methods
- EGNNEditor: Implementation of the basic EGNN editing method
- SEEDGNNEditor: Implementation of the SEED-GNN editing method with mixup training
- Utils: Shared utilities for model loading, validation, testing, and visualization

The framework is designed to be extensible, allowing easy implementation of new editing methods
by subclassing BaseEditor and implementing the edit_model method.

Example usage:
    from editing_pipelines import EGNNEditor, SEEDGNNEditor
    
    # Load configuration
    config = load_config("path/to/config.json")
    
    # Create editor
    editor = EGNNEditor(config)
    
    # Run editing experiment
    raw_results, processed_results = editor.run_editing_experiment()
"""

from ._ensure_repo_paths import bootstrap

bootstrap()

from editing_pipelines.editors.base import BaseEditor
from editing_pipelines.editors.egnn import EGNNEditor
from editing_pipelines.editors.seed_gnn import SEEDGNNEditor
from editing_pipelines.editors.ewc import EWCEditor
from editing_pipelines.editors.multilayer_hypereditor import HyperEditor as HyperGNNEditor
from editing_pipelines.editors.leastsquareseditor import LeastSquaresEditor 
from editing_pipelines.editors.finetuneeditor import FinetuneEditor
from editing_pipelines.utils.model_io import load_model, get_optimizer
from editing_pipelines.utils.train_eval import test, finetune_gnn_mlp
from editing_pipelines.utils.editing_ops import edit
from editing_pipelines.utils.visualization import visualize_validation
from editing_pipelines.utils.results import process_edit_results, process_raw_exp_results
from editing_pipelines.utils.corruption import corrupt_features

__version__ = "1.0.0"
__author__ = "aastha"

# Export main classes and functions
__all__ = [
    # Main editor classes
    'BaseEditor',
    'EGNNEditor', 
    'SEEDGNNEditor',
    'EWCEditor',
    'HyperGNNEditor',
    'LeastSquaresEditor',
    'FinetuneEditor',
    # Utility functions
    'load_model',
    'get_optimizer',
    'test',
    'finetune_gnn_mlp',
    'edit',
    'visualize_validation',
    'process_edit_results',
    'process_raw_exp_results',
    'corrupt_features',
    
]


def create_editor(method: str, config: dict):
    """
    Factory function to create editor instances.
    
    Args:
        method: Name of the editing method ('egnn' or 'seed_gnn')
        config: Configuration dictionary
        
    Returns:
        Editor instance
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == 'egnn':
        return EGNNEditor(config)
    elif method == 'seed_gnn':
        return SEEDGNNEditor(config)
    elif method == 'ewc':
        return EWCEditor(config)
    elif method in ('hyper_gnn', 'hyper'):
        return HyperGNNEditor(config)
    elif method == 'leastsquares':                    
        return LeastSquaresEditor(config)
    elif method == 'finetune':
        return FinetuneEditor(config)
    else:
        raise ValueError(f"Unsupported editing method: {method}. "
                        f"Supported methods: 'egnn', 'seed_gnn', 'ewc', 'hyper_gnn', 'leastsquares'")


def get_available_methods():
    """
    Get list of available editing methods.
    
    Returns:
        List of method names
    """
    return ['egnn', 'seed_gnn', 'ewc', 'hyper_gnn', 'leastsquares', 'finetune']


def get_method_description(method: str) -> str:
    """
    Get description of an editing method.
    
    Args:
        method: Name of the editing method
        
    Returns:
        Description string
        
    Raises:
        ValueError: If method is not supported
    """
    method = method.lower()
    
    if method == 'egnn':
        return "EGNN (Edit GNN) - Independent node editing using gradient-based optimization"
    elif method == 'seed_gnn':
        return ("SEED-GNN (Structured Editing with Neighborhood Mixup) - "
                "Uses mixup training with neighborhood nodes to improve editing performance")
    elif method == 'ewc':
        return ("EWC (Elastic Weight Consolidation) - "
                "Uses EWC regularization to prevent catastrophic forgetting")
    elif method in ('hyper_gnn', 'hyper'):
        return ("HyperGNN (Multi-layer MALMEN HyperEditor) - "
                "Learns hypernetworks to generate layer-wise low-rank edits")
    elif method == 'leastsquares':
        return ("LeastSquares (Least Squares Editing) - "
                "Uses least squares optimization to edit the model")
    elif method == 'finetune':
        return ("Finetune (Fine-tuning Baseline) - "
                "Standard gradient-based fine-tuning on selected layers")
    else:
        raise ValueError(f"Unsupported editing method: {method}. "
                        f"Supported methods: 'egnn', 'seed_gnn', 'ewc', 'hyper_gnn', 'leastsquares'")

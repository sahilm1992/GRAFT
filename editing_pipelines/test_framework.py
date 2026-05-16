#!/usr/bin/env python3
"""
Test script for the editing pipelines framework.

This script demonstrates how to use the editing pipelines framework
to run editing experiments with different methods.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the editing_pipelines to the path
sys.path.append('/home/model_editing/gnn-editing-exploration')
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')

from editing_pipelines import EGNNEditor, SEEDGNNEditor, HyperGNNEditor, create_editor, get_available_methods

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_example_config():
    """Load an example configuration for testing."""
    # Create a minimal config for testing
    config = {
        'management': {
            'dataset_dir': '/home/model_editing/gnn-editing-exploration/seed-gnn/data',
            'pretrain_output_dir': '/home/model_editing/gnn-editing-exploration/seed-gnn/pretrained_models',
            'output_folder_dir': '/home/model_editing/gnn-editing-exploration/editing_pipelines/output'
        },
        'eval_params': {
            'dataset': 'cora',
            'num_targets': 5,
            'subgroup_features': []
        },
        'pipeline_params': {
            'model_name': 'GCN',
            'optim': 'adam',
            'pretrain_lr': 0.01,
            'edit_lr': 0.01,
            'max_num_edit_steps': 10,
            'architecture': {
                'hidden_channels': 64,
                'num_layers': 2,
                'dropout': 0.5
            },
            'load_pretrained_backbone': True,
            # SEED-GNN specific parameters
            'alpha': 0.5,
            'beta': 10
        },
        'corruption': {
            'features': []
        }
    }
    return config


def test_editor_creation():
    """Test creating different editor instances."""
    logger.info("Testing editor creation...")
    
    config = load_example_config()
    
    # Test EGNN editor
    try:
        egnn_editor = EGNNEditor(config)
        logger.info("✓ EGNN editor created successfully")
        logger.info(f"  Method: {egnn_editor.get_method_name()}")
        logger.info(f"  Description: {egnn_editor.get_description()}")
    except Exception as e:
        logger.error(f"✗ Failed to create EGNN editor: {e}")
    
    # Test SEED-GNN editor
    try:
        seed_editor = SEEDGNNEditor(config)
        logger.info("✓ SEED-GNN editor created successfully")
        logger.info(f"  Method: {seed_editor.get_method_name()}")
        logger.info(f"  Description: {seed_editor.get_description()}")
        logger.info(f"  Parameters: {seed_editor.get_parameters()}")
    except Exception as e:
        logger.error(f"✗ Failed to create SEED-GNN editor: {e}")
    
    # Test factory function
    try:
        editor = create_editor('egnn', config)
        logger.info("✓ Factory function works for EGNN")
        
        editor = create_editor('seed_gnn', config)
        logger.info("✓ Factory function works for SEED-GNN")
        editor = create_editor('hyper_gnn', config)
        logger.info("✓ Factory function works for Hyper-GNN")
    except Exception as e:
        logger.error(f"✗ Factory function failed: {e}")


def test_available_methods():
    """Test getting available methods."""
    logger.info("Testing available methods...")
    
    methods = get_available_methods()
    logger.info(f"Available methods: {methods}")
    
    for method in methods:
        try:
            from editing_pipelines import get_method_description
            description = get_method_description(method)
            logger.info(f"  {method}: {description}")
        except Exception as e:
            logger.error(f"✗ Failed to get description for {method}: {e}")


def test_model_loading():
    """Test loading a model and data."""
    logger.info("Testing model and data loading...")
    
    config = load_example_config()
    
    try:
        editor = EGNNEditor(config)
        model, train_data, whole_data, num_features, num_classes = editor.load_model_and_data()
        
        logger.info("✓ Model and data loaded successfully")
        logger.info(f"  Model: {type(model).__name__}")
        logger.info(f"  Train data nodes: {train_data.num_nodes}")
        logger.info(f"  Whole data nodes: {whole_data.num_nodes}")
        logger.info(f"  Features: {num_features}")
        logger.info(f"  Classes: {num_classes}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load model and data: {e}")
        return False


def test_evaluation():
    """Test model evaluation."""
    logger.info("Testing model evaluation...")
    
    config = load_example_config()
    
    try:
        editor = EGNNEditor(config)
        editor.load_model_and_data()
        
        # Test before-edit evaluation
        bef_results = editor.evaluate_before_edit()
        logger.info("✓ Before-edit evaluation successful")
        logger.info(f"  Train acc: {bef_results[0]:.4f}")
        logger.info(f"  Valid acc: {bef_results[1]:.4f}")
        logger.info(f"  Test acc: {bef_results[2]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to evaluate model: {e}")
        return False


def test_edit_target_selection():
    """Test selecting edit targets."""
    logger.info("Testing edit target selection...")
    
    config = load_example_config()
    
    try:
        editor = EGNNEditor(config)
        editor.load_model_and_data()
        
        # Test target selection
        node_idx, labels = editor.select_edit_targets(num_targets=3)
        logger.info("✓ Edit target selection successful")
        logger.info(f"  Selected {len(node_idx)} targets")
        logger.info(f"  Target indices: {node_idx.cpu().numpy()}")
        logger.info(f"  Target labels: {labels.cpu().numpy()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to select edit targets: {e}")
        return False


def run_simple_editing_experiment():
    """Run a simple editing experiment."""
    logger.info("Running simple editing experiment...")
    
    config = load_example_config()
    
    try:
        # Use EGNN for a quick test
        editor = EGNNEditor(config)
        
        # Run the experiment
        raw_results, processed_results = editor.run_editing_experiment(
            num_targets=2,  # Small number for quick test
            max_num_step=5  # Small number for quick test
        )
        
        logger.info("✓ Editing experiment completed successfully")
        logger.info(f"  Raw results: {len(raw_results)} edits")
        logger.info(f"  Processed results keys: {list(processed_results.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Editing experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting editing pipelines framework tests...")
    
    # Test basic functionality
    test_editor_creation()
    test_available_methods()
    
    # Test model operations
    if test_model_loading():
        test_evaluation()
        test_edit_target_selection()
        
        # Only run full experiment if basic tests pass
        logger.info("Basic tests passed. Running simple editing experiment...")
        run_simple_editing_experiment()
    
    logger.info("Framework tests completed!")


if __name__ == "__main__":
    main()

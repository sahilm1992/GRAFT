import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.sparse import coo_matrix
import os

# If you're using torch_geometric Data objects, import Data for typing (optional)
try:
    from torch_geometric.data import Data
except Exception:
    Data = object  # fallback for typing when torch_geometric is not installed


class GNNEdgeAttributePatcher:
    """
    Edge Attribute Patcher for Graph Neural Networks - Node Classification
    
    Implements the EAP method for discovering computational circuits in GNNs by:
    1. Corrupting node features (binary flip, non-binary shuffle)
    2. Computing gradients w.r.t. ground truth class logits
    3. Using first-order Taylor approximation: gradient · (v_corrupted - v_clean)
    """
    
    def __init__(self, model, device: str):
        """
        Initialize the patcher for GNN models
        
        Args:
            model: Trained GNN model (GCN or GCN_MLP)
            device: Device to run computations on
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Storage for activations and gradients
        self.clean_activations = {}
        self.corrupted_activations = {}
        self.gradient_cache = {}
        self.hooks = []
        
        # Get all computation nodes (GCN layers + MLP layers if present)
        self.comp_nodes = self._get_all_comp_nodes()
        
    def _get_all_comp_nodes(self) -> List[str]:
        """Get all computational nodes in the GNN"""
        comp_nodes = []
        
        # Add convolution layers for different backbones
        # Prefer explicit container names (GCN/GAT/GIN/SAGE) but fall back to top-level convs
        for container in ['GCN', 'GAT', 'GIN', 'SAGE', 'Polynormer']:
            if hasattr(self.model, container) and hasattr(getattr(self.model, container), 'convs'):
                convs = getattr(getattr(self.model, container), 'convs')
                for i, _ in enumerate(convs):
                    comp_nodes.append(f"{container}.convs.{i}")
        # Fallback: some models expose convs at top-level
        if hasattr(self.model, 'convs'):
            for i, _ in enumerate(self.model.convs):
                comp_nodes.append(f"convs.{i}")
                
        # Add MLP layers if present
        if hasattr(self.model, 'MLP') and hasattr(self.model.MLP, 'lins'):
            for i, _ in enumerate(self.model.MLP.lins):
                comp_nodes.append(f"MLP.lins.{i}")
                
        return comp_nodes
    
    def _activation_hook(self, name: str):
        """Create hook function to cache activations"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Handle potential tuple outputs
            self.activation_cache[name] = output.detach().clone()
        return hook
    
    def _register_hooks(self):
        """Register forward hooks for all computational nodes"""
        self.hooks = []
        print(f"  [DEBUG] Registering hooks for {len(self.comp_nodes)} computational nodes...")
        
        for name in self.comp_nodes:
            try:
                module = self.model.get_submodule(name)
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
                print(f"  [DEBUG] Hook registered for {name}: {type(module)}")
            except AttributeError:
                print(f"  [DEBUG] WARNING: Could not find module {name}")
                continue

    def _get_module_primary_weight(self, module):
        """Best-effort to find a representative weight tensor for conv/linear modules.
        Handles GCNConv, GATConv, SAGEConv, GINConv and plain Linear.
        Returns a tuple (weight_param, debug_note) where weight_param may be None.
        """
        # Direct weight on module
        if hasattr(module, 'weight') and getattr(module, 'weight') is not None:
            return module.weight, 'module.weight'
        # Common pattern: module.lin.weight (e.g., GCNConv)
        if hasattr(module, 'lin') and hasattr(module.lin, 'weight'):
            return module.lin.weight, 'module.lin.weight'
        # GATConv variations: lin_src/lin_dst or lin_l/lin_r across PyG versions
        for attr in ['lin_src', 'lin_dst', 'lin_l', 'lin_r']:
            if hasattr(module, attr):
                lin_mod = getattr(module, attr)
                if hasattr(lin_mod, 'weight') and lin_mod.weight is not None:
                    return lin_mod.weight, f'module.{attr}.weight'
        # SAGEConv variations: lin_l/lin_r
        for attr in ['lin_l', 'lin_r']:
            if hasattr(module, attr):
                lin_mod = getattr(module, attr)
                if hasattr(lin_mod, 'weight') and lin_mod.weight is not None:
                    return lin_mod.weight, f'module.{attr}.weight'
        # GINConv often wraps an nn Sequential under `nn`; find first Linear
        if hasattr(module, 'nn') and module.nn is not None:
            for sub in module.nn.modules():
                if isinstance(sub, torch.nn.Linear) and hasattr(sub, 'weight'):
                    return sub.weight, 'module.nn.Linear.weight'
        # Fallback: search for the first 2D weight-like parameter
        for name, param in module.named_parameters():
            if 'weight' in name and param is not None and param.dim() >= 2:
                return param, f'named_param:{name}'
        return None, 'no_weight_found'
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def create_feature_corruption(self, clean_data: Data, corrupt_features: Union[List[int], List[str]], 
                                feature_names: Optional[List[str]] = None) -> Data:
        """
        Create corrupted version by modifying specific features
        
        Args:
            clean_data: Clean graph data
            corrupt_features: List of feature indices or names to corrupt
            feature_names: Optional list of feature names for string-based corruption
            
        Returns:
            Corrupted graph data
        """
        print(f"  [DEBUG] Creating corruption for features: {corrupt_features}")
        corrupted_data = clean_data.clone()
        
        # Convert feature names to indices if needed
        if feature_names is not None and len(corrupt_features) > 0 and isinstance(corrupt_features[0], str):
            feat_indices = []
            for feat_name in corrupt_features:
                if feat_name in feature_names:
                    idx = feature_names.index(feat_name)
                    feat_indices.append(idx)
                    print(f"  [DEBUG] Feature '{feat_name}' -> index {idx}")
                else:
                    print(f"  [DEBUG] WARNING: Feature '{feat_name}' not found in feature_names")
        else:
            feat_indices = corrupt_features
            print(f"  [DEBUG] Using feature indices directly: {feat_indices}")
        
        for feat_idx in feat_indices:
            if feat_idx >= clean_data.x.shape[1]:
                print(f"  [DEBUG] WARNING: Feature index {feat_idx} >= num_features {clean_data.x.shape[1]}")
                continue
                
            feature_values = clean_data.x[:, feat_idx]
            unique_vals = torch.unique(feature_values)
            
            print(f"  [DEBUG] Feature {feat_idx}: unique_vals={unique_vals.tolist()}, shape={feature_values.shape}")
            
            # Check if feature is binary (only contains 0 and 1)
            is_binary = (len(unique_vals) <= 2 and 
                        torch.all((unique_vals == 0) | (unique_vals == 1)))
            
            if is_binary:
                # Flip binary features: 0 -> 1, 1 -> 0
                print(f"  [DEBUG] Feature {feat_idx} is binary - flipping values")
                original_mean = feature_values.mean().item()
                corrupted_data.x[:, feat_idx] = 1.0 - feature_values
                new_mean = corrupted_data.x[:, feat_idx].mean().item()
                print(f"  [DEBUG] Binary flip: {original_mean:.3f} -> {new_mean:.3f}")
            else:
                # Shuffle non-binary features across nodes
                print(f"  [DEBUG] Feature {feat_idx} is continuous - shuffling values")
                original_mean = feature_values.mean().item()
                original_std = feature_values.std().item()
                shuffled_indices = torch.randperm(feature_values.size(0))
                corrupted_data.x[:, feat_idx] = feature_values[shuffled_indices]
                new_mean = corrupted_data.x[:, feat_idx].mean().item()
                new_std = corrupted_data.x[:, feat_idx].std().item()
                print(f"  [DEBUG] Shuffle: mean {original_mean:.3f}->{new_mean:.3f}, std {original_std:.3f}->{new_std:.3f}")
        
        return corrupted_data
    
    def get_activation(self, clean_data: Data, corrupted_data: Data) -> torch.Tensor:
        """
        Get activations for clean and corrupted graph inputs
        
        Args:
            clean_data: Clean graph data
            corrupted_data: Corrupted graph data 
            
        Returns:
            Clean model outputs for gradient computation
        """
        print(f"  [DEBUG] Getting activations - registering hooks for {len(self.comp_nodes)} nodes")
        
        # Temporarily unfreeze MLP to ensure its layers execute and hooks fire
        prev_mlp_freezed = getattr(self.model, 'mlp_freezed', None)
        if prev_mlp_freezed is not None and prev_mlp_freezed:
            print(f"  [DEBUG] Temporarily disabling mlp_freezed for activation collection")
            self.model.mlp_freezed = False

        # First, collect activations for clean input (no grad needed for storage)
        self.activation_cache = {}
        self._register_hooks()
        
        clean_data = clean_data.to(self.device)
        with torch.no_grad():
            if hasattr(clean_data, 'adj_t'):
                clean_out = self.model(clean_data.x, clean_data.adj_t)
            else:
                adj_t = SparseTensor.from_edge_index(clean_data.edge_index, 
                                                   sparse_sizes=(clean_data.num_nodes, clean_data.num_nodes))
                clean_out = self.model(clean_data.x, adj_t)
        
        self.clean_activations = {k: v.clone() for k, v in self.activation_cache.items()}
        print(f"  [DEBUG] Clean activations collected: {list(self.clean_activations.keys())}")
        for k, v in self.clean_activations.items():
            print(f"  [DEBUG]   {k}: shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
        
        # Collect activations for corrupted input
        self.activation_cache = {}
        corrupted_data = corrupted_data.to(self.device)
        with torch.no_grad():
            if hasattr(corrupted_data, 'adj_t'):
                corrupt_out = self.model(corrupted_data.x, corrupted_data.adj_t)
            else:
                adj_t = SparseTensor.from_edge_index(corrupted_data.edge_index,
                                                   sparse_sizes=(corrupted_data.num_nodes, corrupted_data.num_nodes))
                corrupt_out = self.model(corrupted_data.x, adj_t)
        
        self.corrupted_activations = {k: v.clone() for k, v in self.activation_cache.items()}
        print(f"  [DEBUG] Corrupted activations collected: {list(self.corrupted_activations.keys())}")
        for k, v in self.corrupted_activations.items():
            print(f"  [DEBUG]   {k}: shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
            if k in self.clean_activations:
                diff = (v - self.clean_activations[k]).abs().mean().item()
                print(f"  [DEBUG]   {k}: avg_abs_diff={diff:.6f}")
        
        self._remove_hooks()
        
        # CRITICAL FIX: Enable gradients for model parameters before forward pass
        print(f"  [DEBUG] Enabling gradients for all model parameters...")
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        # Now return clean outputs WITH gradients enabled for backward pass
        print(f"  [DEBUG] Switching model to train mode for gradient computation")
        self.model.train()  # Enable gradients
        clean_data = clean_data.to(self.device)
        
        # Ensure input requires grad
        if not clean_data.x.requires_grad:
            clean_data.x.requires_grad_(True)
            print(f"  [DEBUG] Set input requires_grad=True")
            
        if hasattr(clean_data, 'adj_t'):
            clean_outputs = self.model(clean_data.x, clean_data.adj_t)
        else:
            adj_t = SparseTensor.from_edge_index(clean_data.edge_index,
                                               sparse_sizes=(clean_data.num_nodes, clean_data.num_nodes))
            clean_outputs = self.model(clean_data.x, adj_t)
        
        print(f"  [DEBUG] Final clean outputs: shape={clean_outputs.shape}, requires_grad={clean_outputs.requires_grad}")
        print(f"  [DEBUG] Output stats: mean={clean_outputs.mean().item():.4f}, std={clean_outputs.std().item():.4f}")
        
        # Verify parameters now require gradients
        params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"  [DEBUG] Parameters requiring gradients after enabling: {params_with_grad}")
        
        self.model.eval()  # Set back to eval mode

        # Restore previous mlp_freezed state after both activation collection and grad-enabled forward
        if prev_mlp_freezed is not None:
            self.model.mlp_freezed = prev_mlp_freezed
            print(f"  [DEBUG] Restored mlp_freezed to {prev_mlp_freezed}")
        return clean_outputs
    
    def get_gradient(self, clean_outputs: torch.Tensor, target_nodes: torch.Tensor, 
                    target_labels: torch.Tensor):
        """
        Compute gradients w.r.t. ground truth class logits (following original EAP)
        
        This follows the original EAP approach of computing gradients with respect to
        the ground truth token/class logit rather than using cross-entropy loss.
        
        Args:
            clean_outputs: Model outputs [num_nodes, num_classes]
            target_nodes: Target node indices [num_target_nodes]
            target_labels: Ground truth labels [num_target_nodes]
        """
        print(f"  [DEBUG] Computing gradients for {len(target_nodes)} target nodes")
        
        # Check if model parameters require gradients
        params_requiring_grad = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_requiring_grad.append(name)
        print(f"  [DEBUG] Parameters requiring gradients: {len(params_requiring_grad)}")
        if len(params_requiring_grad) == 0:
            print(f"  [DEBUG] WARNING: No model parameters require gradients!")
        else:
            print(f"  [DEBUG] Sample params with grad: {params_requiring_grad[:3]}")
        
        # Extract ground truth logits for target nodes (following original EAP)
        target_logits = []
        
        for i, (node_idx, label) in enumerate(zip(target_nodes, target_labels)):
            # Ensure indices are within bounds
            if node_idx >= clean_outputs.shape[0]:
                print(f"  [DEBUG] WARNING: Node index {node_idx} >= num_nodes {clean_outputs.shape[0]}")
                continue
            if label >= clean_outputs.shape[1]:
                print(f"  [DEBUG] WARNING: Label {label} >= num_classes {clean_outputs.shape[1]}")
                continue
                
            # Get the logit for the ground truth class for this target node
            ground_truth_logit = clean_outputs[node_idx, label]
            target_logits.append(ground_truth_logit)
            
            # Show top-3 predictions for context
            num_classes = clean_outputs.shape[1]
            k = min(3, num_classes)  # Fix topk error
            if k > 0:
                top_values, top_indices = clean_outputs[node_idx].topk(k)
        
        # Sum the ground truth logits (similar to original approach)
        if len(target_logits) > 0:
            total_ground_truth_logit = torch.stack(target_logits).sum()
            print(f"  [DEBUG] Total ground truth logit: {total_ground_truth_logit.item():.4f}")
            print(f"  [DEBUG] Total logit requires_grad: {total_ground_truth_logit.requires_grad}")
        else:
            # Fallback if no target nodes
            total_ground_truth_logit = clean_outputs.sum()
            print(f"  [DEBUG] Fallback: using sum of all outputs: {total_ground_truth_logit.item():.4f}")
        
        if not total_ground_truth_logit.requires_grad:
            print(f"  [DEBUG] ERROR: total_ground_truth_logit does not require gradients!")
            print(f"  [DEBUG] clean_outputs.requires_grad: {clean_outputs.requires_grad}")
            return
        
        print(f"  [DEBUG] Starting backward pass...")
        # Backward pass to compute gradients w.r.t. ground truth logits
        total_ground_truth_logit.backward()
        
        # Cache gradients from all computational layers
        gradient_stats = {}
        print(f"  [DEBUG] Checking gradients in {len(self.comp_nodes)} computational nodes...")
        
        for name in self.comp_nodes:
            print(f"  [DEBUG] Checking module: {name}")
            try:
                module = self.model.get_submodule(name)
                print(f"  [DEBUG]   Module found: {type(module)}")
                
                weight_param, found_note = self._get_module_primary_weight(module)
                if weight_param is None:
                    print(f"  [DEBUG]   No suitable weight found ({found_note})")
                    # List some attributes for debugging
                    attrs = [attr for attr in dir(module) if not attr.startswith('_') and not callable(getattr(module, attr))]
                    print(f"  [DEBUG]   Available non-callable attributes: {attrs[:10]}...")
                    continue
                else:
                    print(f"  [DEBUG]   Using weight from {found_note}: shape={tuple(weight_param.shape)}")
                
                if weight_param.grad is not None:
                    grad = weight_param.grad.detach().clone()
                    self.gradient_cache[name] = grad
                    gradient_stats[name] = {
                        'shape': grad.shape,
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max_abs': grad.abs().max().item()
                    }
                    print(f"  [DEBUG]   Gradient cached: shape={grad.shape}, mean={grad.mean().item():.6f}, max_abs={grad.abs().max().item():.6f}")
                else:
                    print(f"  [DEBUG]   Weight grad is None")
                    print(f"  [DEBUG]   Weight requires_grad: {weight_param.requires_grad}")
                    
            except AttributeError as e:
                print(f"  [DEBUG]   AttributeError: {e}")
                continue
            except Exception as e:
                print(f"  [DEBUG]   Unexpected error: {e}")
                continue
        
        print(f"  [DEBUG] Cached gradients for {len(self.gradient_cache)}/{len(self.comp_nodes)} modules")
        
        # Clear gradients
        self.model.zero_grad()
        print(f"  [DEBUG] Cleared model gradients")
    
    def get_causal_effect(self, comp_node: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal effect using Edge Attribute Patching
        
        Core EAP formula: gradient · (v_corrupted - v_clean)
        This gives the predicted change in ground truth logit due to the corruption.
        
        Args:
            comp_node: Name of computational node
            
        Returns:
            Tuple of (causal_effect per node, clean_activation)
        """
        if comp_node not in self.clean_activations or comp_node not in self.corrupted_activations:
            print(f"  [DEBUG] Missing activations for {comp_node}")
            return torch.tensor(0.0), torch.tensor(0.0)
        
        clean_activation = self.clean_activations[comp_node]      # [num_nodes, hidden_dim]
        corrupted_activation = self.corrupted_activations[comp_node]  # [num_nodes, hidden_dim]
        
        # Compute activation difference (v_corrupted - v_clean)
        activation_diff = corrupted_activation - clean_activation  # [num_nodes, hidden_dim]
        
        print(f"  [DEBUG] {comp_node} activation diff: shape={activation_diff.shape}")
        print(f"  [DEBUG]   mean_abs_diff={activation_diff.abs().mean().item():.6f}")
        print(f"  [DEBUG]   max_abs_diff={activation_diff.abs().max().item():.6f}")
        print(f"  [DEBUG]   nodes_with_change={torch.sum(activation_diff.abs().sum(dim=-1) > 1e-6).item()}")
        
        # Get corresponding gradient
        if comp_node not in self.gradient_cache:
            print(f"  [DEBUG] No gradient cached for {comp_node}")
            return torch.tensor(0.0), clean_activation
            
        gradient = self.gradient_cache[comp_node]  # [input_dim, output_dim] or similar
        print(f"  [DEBUG] {comp_node} gradient shape: {gradient.shape}")
        
        # Compute causal effect: activation_diff · gradient
        # This predicts how the corruption affects the ground truth logit
        if len(gradient.shape) == 2:
            # Standard linear layer: activation_diff @ gradient
            print(f"  [DEBUG] Using matrix multiplication: activation_diff @ gradient")
            try:
                print(f"  [DEBUG] Activation diff shape: {activation_diff.shape}")
                print(f"  [DEBUG] Gradient shape: {gradient.shape}")
                causal_effect = torch.sum(activation_diff @ gradient, dim=-1)  # [num_nodes]
                print(f"  [DEBUG] Causal effect computed successfully")
                print(f"  [DEBUG] Causal effect shape: {causal_effect.shape}")
            except RuntimeError as e:
                print(f"  [DEBUG] Matrix multiplication failed: {e}")
                # Fallback to element-wise
                if gradient.shape[0] == activation_diff.shape[-1]:
                    causal_effect = torch.sum(activation_diff * gradient.mean(dim=-1, keepdim=True), dim=-1)
                else:
                    causal_effect = torch.sum(activation_diff * gradient.mean(), dim=-1)
        else:
            # Handle other cases - element-wise multiplication
            print(f"  [DEBUG] Using element-wise multiplication")
            # Reduce gradient to match activation dimensions if needed
            if gradient.numel() > activation_diff.numel():
                gradient_reduced = gradient.mean(dim=-1, keepdim=True)
                print(f"  [DEBUG] Reduced gradient shape: {gradient_reduced.shape}")
            else:
                gradient_reduced = gradient
            causal_effect = torch.sum(activation_diff * gradient_reduced, dim=-1)
        
        print(f"  [DEBUG] Final causal effect: shape={causal_effect.shape}")
        print(f"  [DEBUG]   mean_abs={causal_effect.abs().mean().item():.6f}")
        print(f"  [DEBUG]   max_abs={causal_effect.abs().max().item():.6f}")
        print(f"  [DEBUG]   nonzero_effects={torch.sum(causal_effect.abs() > 1e-8).item()}")
        
        return causal_effect, clean_activation
    
    def find_node_importance(self, data_pairs: List[Tuple[Data, Union[List[int], List[str]], torch.Tensor, torch.Tensor]], 
                           feature_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Find important computational nodes using causal intervention
        
        Args:
            data_pairs: List of (graph_data, corrupt_features, target_nodes, target_labels)
            feature_names: Optional list of feature names for string-based corruption
            
        Returns:
            Dictionary mapping computational node names to importance scores [num_graph_nodes]
        """
        effects = {node: [] for node in self.comp_nodes}
        
        print(f"\n=== Starting Circuit Discovery ===")
        print(f"Computational nodes found: {self.comp_nodes}")
        print(f"Processing {len(data_pairs)} corruption experiments...")
        
        for i, (graph_data, corrupt_features, target_nodes, target_labels) in enumerate(tqdm(data_pairs, desc="Processing experiments")):
            try:
                print(f"\nExperiment {i+1}/{len(data_pairs)}:")
                print(f"  Corrupting features: {corrupt_features}")
                
                # Create corrupted version
                corrupted_data = self.create_feature_corruption(graph_data, corrupt_features, feature_names)
                
                # Get activations
                clean_outputs = self.get_activation(graph_data, corrupted_data)
                print(f"  Clean outputs shape: {clean_outputs.shape}")
                
                # Compute gradients w.r.t. ground truth logits
                self.get_gradient(clean_outputs, target_nodes, target_labels)
                print(f"  Gradients computed for {len(self.gradient_cache)} modules")
                
                # Compute causal effects for all computational nodes
                exp_effects = {}
                for node in self.comp_nodes:
                    effect, _ = self.get_causal_effect(node)
                    effects[node].append(effect)
                    exp_effects[node] = effect
                    print(f"  {node}: max_effect={effect.abs().max().item():.6f}, mean_effect={effect.abs().mean().item():.6f}")
                
                # Cleanup
                del clean_outputs, corrupted_data
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ERROR in experiment {i+1}: {e}")
                continue
        
        # Average absolute effects across all corruption experiments
        print(f"\n=== Computing Final Importance Scores ===")
        node_importance = {}
        for node, effect_list in effects.items():
            if len(effect_list) > 0:
                stacked_effects = torch.stack(effect_list, dim=0)  # [num_experiments, num_nodes]
                node_importance[node] = torch.mean(torch.abs(stacked_effects), dim=0)  # [num_nodes]
                print(f"{node}:")
                print(f"  Processed {len(effect_list)} experiments")
                print(f"  Final importance - max: {node_importance[node].max().item():.6f}")
                print(f"  Final importance - mean: {node_importance[node].mean().item():.6f}")
                # Fix the len() error for topk
                if node_importance[node].numel() > 0:
                    k = min(5, node_importance[node].numel())
                    if k > 0:
                        top_nodes = node_importance[node].topk(k).indices.tolist()
                        print(f"  Top-{k} nodes: {top_nodes}")
                    else:
                        print(f"  No nodes to show (numel={node_importance[node].numel()})")
                else:
                    print(f"  Empty tensor - no nodes to show")
            else:
                node_importance[node] = torch.zeros(1)
                print(f"{node}: No valid experiments - set to zero importance")
        
        return node_importance
    
    def find_circuit_nodes(self, node_importance: Dict[str, torch.Tensor], 
                          topk: int = 10) -> List[Tuple[str, int]]:
        """
        Find top-k most important (computational_node, graph_node) pairs
        
        Args:
            node_importance: Dictionary of node importance scores
            topk: Number of top (comp_node, graph_node) pairs to select
            
        Returns:
            List of (computational_node_name, graph_node_index) tuples
        """
        print(f"\n=== Selecting Top-{topk} Circuit Components ===")
        
        all_scores = []
        all_identifiers = []
        
        for comp_node_name, scores in node_importance.items():
            for graph_node_idx, score in enumerate(scores.flatten()):
                all_scores.append(score.item())
                all_identifiers.append((comp_node_name, graph_node_idx))
        
        print(f"Total candidates: {len(all_scores)}")
        
        # Select top-k most important
        if len(all_scores) == 0:
            print("No scores found - returning empty list")
            return []
        
        top_indices = torch.tensor(all_scores).topk(min(topk, len(all_scores))).indices
        selected_nodes = [all_identifiers[i] for i in top_indices]
        selected_scores = [all_scores[i] for i in top_indices]
        
        print("Selected circuit components:")
        for i, ((comp_node, graph_node), score) in enumerate(zip(selected_nodes, selected_scores)):
            print(f"  {i+1:2d}. {comp_node:<15} | Node {graph_node:3d} | Score: {score:.6f}")
        
        return selected_nodes
    
    def reset(self):
        """Reset all cached data and free memory"""
        self.clean_activations.clear()
        self.corrupted_activations.clear()
        self.gradient_cache.clear()
        if hasattr(self, 'activation_cache'):
            self.activation_cache.clear()
        torch.cuda.empty_cache()


class GNNCircuitVisualizer:
    """Lightweight visualization for GNN circuit discovery - neighbor analysis only"""

    def __init__(self):
        pass

    def create_networkx_graph(self, graph_data: Data) -> nx.Graph:
        """Convert PyTorch Geometric-like data to NetworkX graph with robust error handling"""
        G = nx.Graph()

        # Add nodes: try common attributes in order
        num_nodes = None
        if hasattr(graph_data, 'num_nodes') and graph_data.num_nodes is not None:
            try:
                num_nodes = int(graph_data.num_nodes)
            except Exception:
                pass
        if num_nodes is None:
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                try:
                    num_nodes = int(graph_data.x.shape[0])
                except Exception:
                    pass
        if num_nodes is None:
            # fallback: check edge_index or adjacency to infer node ids
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                ei = graph_data.edge_index
                try:
                    arr = ei.cpu().numpy()
                    num_nodes = int(arr.max()) + 1
                except Exception:
                    pass
        if num_nodes is None:
            num_nodes = 0

        for i in range(num_nodes):
            G.add_node(i)

        # Add edges - handle different possible formats
        edges = []

        try:
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                # Standard PyTorch Geometric format
                edge_index = graph_data.edge_index
                if hasattr(edge_index, 'cpu'):
                    edge_index = edge_index.cpu().numpy()
                else:
                    edge_index = np.array(edge_index)
                if edge_index.ndim == 2 and edge_index.shape[0] >= 2:
                    edges = [(int(edge_index[0, i]), int(edge_index[1, i]))
                             for i in range(edge_index.shape[1])]
            elif hasattr(graph_data, 'adj_t') and graph_data.adj_t is not None:
                # SparseTensor format (torch_sparse / PyG)
                adj_t = graph_data.adj_t
                # try .coo()
                if hasattr(adj_t, 'coo'):
                    row, col, _ = adj_t.coo()  # many implementations return (row, col, value)
                    # row/col may be tensors or numpy arrays
                    row = row.cpu().numpy() if hasattr(row, 'cpu') else np.array(row)
                    col = col.cpu().numpy() if hasattr(col, 'cpu') else np.array(col)
                    edges = [(int(row[i]), int(col[i])) for i in range(len(row))]
                elif hasattr(adj_t, 'to_edge_index'):
                    ei = adj_t.to_edge_index()
                    # some return (edge_index, e_attr)
                    if isinstance(ei, (list, tuple)):
                        ei = ei[0]
                    ei = ei.cpu().numpy() if hasattr(ei, 'cpu') else np.array(ei)
                    if ei.ndim == 2 and ei.shape[0] >= 2:
                        edges = [(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])]
            elif hasattr(graph_data, 'adjacency_matrix') and graph_data.adjacency_matrix is not None:
                adj_matrix = graph_data.adjacency_matrix
                if hasattr(adj_matrix, 'cpu'):
                    adj_matrix = adj_matrix.cpu().numpy()
                else:
                    adj_matrix = np.array(adj_matrix)
                rows, cols = np.nonzero(adj_matrix)
                edges = [(int(rows[i]), int(cols[i])) for i in range(len(rows))]
        except Exception as e:
            print(f"  [WARNING] Error while extracting edges: {e}")

        if edges:
            G.add_edges_from(edges)
            print(f"  [DEBUG] Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        else:
            print(f"  [WARNING] No edge information found in graph_data, creating graph with isolated nodes")

        return G

    def analyze_circuit_neighborhoods(self, graph_data: Data,
                                    circuit_nodes: List[Tuple[str, int]],
                                    top_k: int = 10,
                                    figsize: Tuple[int, int] = (12, 8),
                                    save_path: Optional[str] = None):
        """
        Analyze and plot neighborhood statistics for the most important circuit nodes
        
        Args:
            graph_data: Graph data object
            circuit_nodes: List of (layer_name, node_idx) pairs for important nodes
            top_k: Number of top nodes to analyze
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        try:
            # Convert to NetworkX
            G = self.create_networkx_graph(graph_data)
            
            # Take only top_k most important nodes
            important_nodes = circuit_nodes[:top_k]
            
            # Collect statistics for each node
            node_stats = []
            
            for layer_name, node_idx in important_nodes:
                if node_idx not in G.nodes():
                    print(f"  [WARNING] Node {node_idx} not found in graph")
                    continue
                    
                # Calculate degree
                degree = G.degree(node_idx)
                
                # Calculate k-hop neighbors for k=1,2,3
                hop_counts = {}
                try:
                    # Get shortest path lengths from this node to all reachable nodes
                    path_lengths = nx.single_source_shortest_path_length(G, node_idx, cutoff=3)
                    
                    # Count nodes at each hop distance
                    for hop in [1, 2, 3]:
                        hop_counts[hop] = sum(1 for d in path_lengths.values() if d == hop)
                        
                except Exception as e:
                    print(f"  [WARNING] Could not compute paths for node {node_idx}: {e}")
                    hop_counts = {1: 0, 2: 0, 3: 0}
                
                node_stats.append({
                    'layer': layer_name,
                    'node': node_idx,
                    'degree': degree,
                    '1_hop': hop_counts[1],
                    '2_hop': hop_counts[2], 
                    '3_hop': hop_counts[3]
                })
            
            if not node_stats:
                print("No valid nodes found for analysis")
                return
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # Extract data for plotting
            nodes = [f"{s['layer'][:8]}_{s['node']}" for s in node_stats]
            degrees = [s['degree'] for s in node_stats]
            hop1 = [s['1_hop'] for s in node_stats]
            hop2 = [s['2_hop'] for s in node_stats]
            hop3 = [s['3_hop'] for s in node_stats]
            
            x_pos = range(len(nodes))
            
            # Plot 1: Degree
            ax1.bar(x_pos, degrees, alpha=0.7, color='steelblue')
            ax1.set_title('Node Degrees')
            ax1.set_ylabel('Degree')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(nodes, rotation=45, ha='right')
            
            # Plot 2: 1-hop neighbors
            ax2.bar(x_pos, hop1, alpha=0.7, color='orange')
            ax2.set_title('1-Hop Neighbors')
            ax2.set_ylabel('Count')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(nodes, rotation=45, ha='right')
            
            # Plot 3: 2-hop neighbors
            ax3.bar(x_pos, hop2, alpha=0.7, color='green')
            ax3.set_title('2-Hop Neighbors')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(nodes, rotation=45, ha='right')
            
            # Plot 4: 3-hop neighbors
            ax4.bar(x_pos, hop3, alpha=0.7, color='red')
            ax4.set_title('3-Hop Neighbors')
            ax4.set_ylabel('Count')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(nodes, rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  [INFO] Saved neighborhood analysis: {save_path}")
            
            plt.show()
            
            # Print summary statistics
            print(f"\n=== Neighborhood Analysis Summary ===")
            print(f"Analyzed top {len(node_stats)} circuit nodes:")
            for stat in node_stats:
                print(f"  {stat['layer'][:12]:<12} Node {stat['node']:3d}: "
                      f"degree={stat['degree']:3d}, 1-hop={stat['1_hop']:3d}, "
                      f"2-hop={stat['2_hop']:3d}, 3-hop={stat['3_hop']:3d}")
            
            return node_stats
            
        except Exception as e:
            print(f"[ERROR] Neighborhood analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def reset(self):
        """Reset any cached data"""
        pass

    def plot_cumulative_importance(self, node_importance: Dict[str, torch.Tensor],
                                   max_points: Optional[int] = None,
                                   figsize: Tuple[int, int] = (12, 8),
                                   save_path: Optional[str] = None):
        """Plot cumulative % of total importance vs number of nodes for each module.
        Args:
            node_importance: dict of {comp_node_name: importance_tensor [num_nodes]}
            max_points: optional cap for x-axis to improve readability
            figsize: figure size
            save_path: optional file to save the plot
        """
        if node_importance is None or len(node_importance) == 0:
            print("[DEBUG] plot_cumulative_importance: empty node_importance; nothing to plot")
            return

        # Compute global total importance across all modules
        global_total = 0.0
        for scores in node_importance.values():
            if scores is not None and scores.numel() > 0:
                global_total += scores.detach().float().cpu().abs().sum().item()
        if global_total <= 0:
            print("[DEBUG] plot_cumulative_importance: non-positive global total importance")
            return

        plt.figure(figsize=figsize)
        for comp_node, scores in node_importance.items():
            if scores is None or scores.numel() == 0:
                continue
            s = scores.detach().float().cpu().abs()
            sorted_vals, _ = torch.sort(s, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=0)
            # Normalize by global total so endpoints across modules sum to 100%
            cumperc = (cumsum / global_total) * 100.0
            x = torch.arange(1, cumperc.numel() + 1)
            if max_points is not None and max_points < x.numel():
                x = x[:max_points]
                cumperc = cumperc[:max_points]
            plt.plot(x.numpy(), cumperc.numpy(), label=comp_node, linewidth=1.5)

        plt.xlabel('Number of nodes')
        plt.ylabel('% of global total importance (cumulative)')
        plt.title('Per-module cumulative coverage of global importance')
        plt.legend(fontsize=8, ncol=2, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [INFO] Saved cumulative importance plot: {save_path}")
        plt.close()

    def plot_global_cumulative_importance(self, node_importance: Dict[str, torch.Tensor],
                                          max_points: Optional[int] = None,
                                          figsize: Tuple[int, int] = (12, 6),
                                          save_path: Optional[str] = None):
        """Aggregate importance across modules per graph node, then sort nodes by
        total importance and plot cumulative % of total vs number of nodes.
        """
        if node_importance is None or len(node_importance) == 0:
            print("[DEBUG] plot_global_cumulative_importance: empty node_importance; nothing to plot")
            return

        # Determine num_nodes as max length among tensors
        num_nodes = 0
        for scores in node_importance.values():
            if scores is not None:
                num_nodes = max(num_nodes, int(scores.numel()))
        if num_nodes == 0:
            print("[DEBUG] plot_global_cumulative_importance: could not infer num_nodes")
            return

        # Sum importance across modules per node (broadcast smaller tensors if needed)
        total_importance = None
        for scores in node_importance.values():
            if scores is None or scores.numel() == 0:
                continue
            s = scores.detach().float().cpu().abs().view(-1)
            if s.numel() < num_nodes:
                # pad with zeros to match length
                pad = torch.zeros(num_nodes - s.numel())
                s = torch.cat([s, pad], dim=0)
            elif s.numel() > num_nodes:
                s = s[:num_nodes]
            total_importance = s if total_importance is None else (total_importance + s)

        if total_importance is None:
            print("[DEBUG] plot_global_cumulative_importance: no valid scores found")
            return

        grand_total = total_importance.sum().item()
        if grand_total <= 0:
            print("[DEBUG] plot_global_cumulative_importance: non-positive total importance")
            return

        sorted_vals, _ = torch.sort(total_importance, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=0)
        cumperc = (cumsum / grand_total) * 100.0
        x = torch.arange(1, cumperc.numel() + 1)
        if max_points is not None and max_points < x.numel():
            x = x[:max_points]
            cumperc = cumperc[:max_points]

        plt.figure(figsize=figsize)
        plt.plot(x.numpy(), cumperc.numpy(), color='black', linewidth=2.0)
        plt.xlabel('Number of nodes (sorted by total importance)')
        plt.ylabel('% of total importance (cumulative)')
        plt.title('Global cumulative importance coverage by top nodes')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [INFO] Saved global cumulative importance plot: {save_path}")
        plt.close()

    def plot_modulewise_sorted_percentages(self, node_importance: Dict[str, torch.Tensor],
                                           max_points_per_module: Optional[int] = None,
                                           figsize: Tuple[int, int] = (12, 8),
                                           save_path: Optional[str] = None):
        """For each module, sort nodes by decreasing importance and plot their
        absolute percentage importances relative to the global total importance.
        """
        if node_importance is None or len(node_importance) == 0:
            print("[DEBUG] plot_modulewise_sorted_percentages: empty node_importance; nothing to plot")
            return

        # Compute global total (sum over abs(module importances))
        global_total = 0.0
        per_module_scores = {}
        for comp_node, scores in node_importance.items():
            if scores is None or scores.numel() == 0:
                continue
            s = scores.detach().float().cpu().abs().view(-1)
            per_module_scores[comp_node] = s
            global_total += s.sum().item()

        if global_total <= 0:
            print("[DEBUG] plot_modulewise_sorted_percentages: non-positive global total importance")
            return

        plt.figure(figsize=figsize)
        for comp_node, s in per_module_scores.items():
            if s.numel() == 0:
                continue
            sorted_vals, _ = torch.sort(s, descending=True)
            perc = (sorted_vals / global_total) * 100.0
            x = torch.arange(1, perc.numel() + 1)
            if max_points_per_module is not None and max_points_per_module < x.numel():
                x = x[:max_points_per_module]
                perc = perc[:max_points_per_module]
            plt.plot(x.numpy(), perc.numpy(), label=comp_node, linewidth=1.5)

        plt.xlabel('Node rank within module (sorted by importance)')
        plt.ylabel('Absolute % of global importance per node')
        plt.title('Module-wise sorted node importances (absolute percentages)')
        plt.legend(fontsize=8, ncol=2, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [INFO] Saved modulewise sorted percentages plot: {save_path}")
        plt.close()


def discover_gnn_circuits(model, graph_data: Data, 
                         target_nodes: torch.Tensor,
                         target_labels: torch.Tensor,
                         features_to_corrupt: List[Union[List[int], List[str]]],
                         feature_names: Optional[List[str]] = None,
                         device: str = 'cuda', topk: int = 10,
                         visualize: bool = True,
                         save_plots: Optional[str] = None):
    """
    Complete pipeline for discovering circuits in GNN for node classification
    
    Args:
        model: Trained GNN model
        graph_data: Graph data object
        target_nodes: Target node indices for classification [num_target_nodes]
        target_labels: Target labels [num_target_nodes]
        features_to_corrupt: List of feature sets to corrupt, can be:
            - [[0], [1], [0,1]] (indices)
            - [["age"], ["income"], ["age", "income"]] (names)
        feature_names: List of feature names for string-based corruption
        device: Device for computation
        topk: Number of top (comp_node, graph_node) pairs to select
        visualize: Whether to create neighborhood analysis
        save_plots: Path to save neighborhood analysis plot
        
    Returns:
        Tuple of (circuit_nodes, node_importance, visualizer):
        - circuit_nodes: List of (computational_node, graph_node) pairs
        - node_importance: Dict mapping comp_node names to importance scores
        - visualizer: GNNCircuitVisualizer instance for additional analysis
    """
    print("="*60)
    print("GNN CIRCUIT DISCOVERY USING EDGE ATTRIBUTE PATCHING")
    print("="*60)
    
    print(f"\nInput Summary:")
    print(f"  Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    print(f"  Features: {graph_data.x.shape[1]} dimensions")
    print(f"  Target nodes: {len(target_nodes)} nodes")
    print(f"  Device: {device}")
    
    if feature_names:
        print(f"  Feature names: {feature_names}")
    
    print(f"\nCorruption Strategy:")
    for i, corrupt_feats in enumerate(features_to_corrupt):
        print(f"  Experiment {i+1}: {corrupt_feats}")
    
    patcher = GNNEdgeAttributePatcher(model, device)
    
    # Create data pairs for different corruption experiments
    data_pairs = []
    for corrupt_features in features_to_corrupt:
        data_pairs.append((graph_data, corrupt_features, target_nodes, target_labels))
    
    # Find node importance using EAP
    node_importance = patcher.find_node_importance(data_pairs, feature_names)
    
    # Select top circuit nodes
    circuit_nodes = patcher.find_circuit_nodes(node_importance, topk)
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    total_importance = sum(scores.sum().item() for scores in node_importance.values())
    print(f"  Total importance across all components: {total_importance:.6f}")
    
    for comp_node, scores in node_importance.items():
        comp_total = scores.sum().item()
        comp_percentage = (comp_total / total_importance * 100) if total_importance > 0 else 0
        print(f"  {comp_node}: {comp_percentage:.2f}% of total importance")
    
    print(f"\nFinal Circuit:")
    print(f"  Selected {len(circuit_nodes)} most important components")
    print(f"  Circuit covers {len(set(node[1] for node in circuit_nodes))} unique graph nodes")
    print(f"  Circuit spans {len(set(node[0] for node in circuit_nodes))} computational layers")
    
    # Create neighborhood analysis
    visualizer = None
    if visualize:
        print(f"\n=== Creating Neighborhood Analysis ===")
        visualizer = GNNCircuitVisualizer()
        
        # Analyze neighborhoods of top circuit nodes
        visualizer.analyze_circuit_neighborhoods(
            graph_data, circuit_nodes, top_k=min(15, len(circuit_nodes)),
            save_path=save_plots
        )
        # Also plot cumulative importance vs number of nodes per module
        try:
            base_dir = os.path.dirname(save_plots) if save_plots else '.'
            cum_path = os.path.join(base_dir, 'cumulative_importance.png')
            visualizer.plot_cumulative_importance(node_importance, max_points=10000, save_path=cum_path)
        except Exception as e:
            print(f"[DEBUG] Failed to save cumulative importance plot: {e}")
        # And global cumulative importance across modules
        try:
            base_dir = os.path.dirname(save_plots) if save_plots else '.'
            global_cum_path = os.path.join(base_dir, 'global_cumulative_importance.png')
            visualizer.plot_global_cumulative_importance(node_importance, max_points=10000, save_path=global_cum_path)
        except Exception as e:
            print(f"[DEBUG] Failed to save global cumulative importance plot: {e}")
        # Module-wise sorted absolute percentage importances
        try:
            base_dir = os.path.dirname(save_plots) if save_plots else '.'
            module_sorted_path = os.path.join(base_dir, 'modulewise_sorted_percentages.png')
            visualizer.plot_modulewise_sorted_percentages(node_importance, max_points_per_module=10000, save_path=module_sorted_path)
        except Exception as e:
            print(f"[DEBUG] Failed to save modulewise sorted percentages plot: {e}")
    
    patcher.reset()
    print("\nCircuit discovery completed!")
    print("="*60)
    
    return circuit_nodes, node_importance, visualizer


# Example usage:
"""
# Using feature indices with neighborhood analysis
features_to_corrupt = [[0], [1], [0, 1], [2, 3]]
circuit_nodes, importance, visualizer = discover_gnn_circuits(
    model=trained_gcn,
    graph_data=my_graph,
    target_nodes=torch.tensor([10, 20, 30]),  # Nodes to classify
    target_labels=torch.tensor([0, 1, 2]),    # Their true classes  
    features_to_corrupt=features_to_corrupt,
    device='cuda',
    topk=15,
    visualize=True,
    save_plots='neighborhood_analysis.png'
)

# Using feature names
feature_names = ["age", "income", "education", "location"]
features_to_corrupt = [["age"], ["income"], ["age", "income"], ["education", "location"]]
circuit_nodes, importance, visualizer = discover_gnn_circuits(
    model=trained_gcn,
    graph_data=my_graph,
    target_nodes=torch.tensor([5, 15, 25, 35]),
    target_labels=torch.tensor([0, 1, 0, 2]),
    features_to_corrupt=features_to_corrupt,
    feature_names=feature_names,
    device='cuda',
    topk=20,
    visualize=True
)

# Additional analysis if needed
node_stats = visualizer.analyze_circuit_neighborhoods(my_graph, circuit_nodes, top_k=10)
"""
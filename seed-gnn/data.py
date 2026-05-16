import logging
logger = logging.getLogger("main")

from typing import Tuple, Union, Optional
import time
import numpy as np
import pandas as pd
import os
import copy
import torch
import scipy.io as sio
import scipy.sparse as sp
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import matplotlib.pyplot as plt
import yaml
from sklearn.preprocessing import OneHotEncoder
# import main_utils
# config = main_utils.register_args_and_configs(args)

def get_device():
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    has_visible_cuda = os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
    if torch.cuda.is_available() and has_visible_cuda and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = (torch.unique(y)).size(0)

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        print(f"Class {c}: {idx.size(0)} samples")
        
        # Handle percentage vs absolute counts
        if train_per_class < 1:
            # Percentage mode - ignore num_splits for percentage calculation
            train_count = int(train_per_class * idx.size(0))
        else:
            # Absolute count mode - use as-is
            train_count = train_per_class
            
        if val_per_class < 1:
            # Percentage mode - ignore num_splits for percentage calculation  
            val_count = int(val_per_class * idx.size(0))
        else:
            # Absolute count mode - use as-is
            val_count = val_per_class

        print(f"Class {c}: train_count={train_count}, val_count={val_count}")
        
        # Ensure we don't exceed available samples
        available_samples = idx.size(0)
        if train_count + val_count > available_samples:
            print(f"Warning: Not enough samples in class {c}. Available: {available_samples}, Requested: {train_count + val_count}")
            train_count = min(train_count, available_samples)
            val_count = min(val_count, available_samples - train_count)
        
        # Simple random permutation
        perm = torch.randperm(idx.size(0))
        idx_shuffled = idx[perm]
        
        # Take the first train_count for training
        if train_count > 0:
            train_idx = idx_shuffled[:train_count]
            train_mask[train_idx] = True
        
        # Take the next val_count for validation
        if val_count > 0:
            val_idx = idx_shuffled[train_count:train_count + val_count]
            val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    # Handle negative indices in bincount by finding min label and adjusting
    min_label = y.min().item()
    if min_label < 0:
        # Shift labels to make them non-negative for bincount
        y_shifted = y - min_label
        train_labels = y_shifted[train_mask].bincount()
        val_labels = y_shifted[val_mask].bincount() if val_mask.sum() > 0 else torch.tensor([])
        test_labels = y_shifted[test_mask].bincount() if test_mask.sum() > 0 else torch.tensor([])
        
        print(f"Train labels (shifted from {min_label}): {train_labels}")
        print(f"Val labels (shifted from {min_label}): {val_labels}")
        print(f"Test labels (shifted from {min_label}): {test_labels}")
    else:
        print("Train labels:", y[train_mask].bincount())
        print("Val labels:", y[val_mask].bincount() if val_mask.sum() > 0 else torch.tensor([]))
        print("Test labels:", y[test_mask].bincount() if test_mask.sum() > 0 else torch.tensor([]))


    return train_mask, val_mask, test_mask

def analyze_feature_correlation(
    data: Data,
    feature_names: Optional[list] = None,
    method: str = "spearman",
    include_labels: bool = False
) -> pd.DataFrame:
    """
    Compute feature correlation matrix for the given dataset.

    Args:
        data (Data): A PyG Data object with `x` containing node features.
        feature_names (list, optional): List of feature names corresponding to columns in `x`.
        method (str): Correlation method ('pearson', 'spearman', or 'kendall').
        include_labels (bool): If True, include target `y` as a column in the correlation analysis.

    Returns:
        pd.DataFrame: Correlation matrix as a pandas DataFrame.
    """
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Data object has no feature matrix `x`.")

    features = data.x.cpu().numpy()

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(features.shape[1])]

    df = pd.DataFrame(features, columns=feature_names)

    if include_labels:
        if data.y is None:
            raise ValueError("Data object has no labels (`y`).")
        df["label"] = data.y.cpu().numpy()

    corr_matrix = df.corr(method=method)
    return corr_matrix

def get_top_correlated_features(corr_matrix: pd.DataFrame, target_features: list, top_k: int = 5) -> dict:
    """
    Find the most correlated features with the given feature names.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix returned from `analyze_feature_correlation`.
        target_features (list): List of feature names to analyze.
        top_k (int): Number of most correlated features to return for each target.

    Returns:
        dict: {feature_name: list of (other_feature, correlation_value)} for each target feature.
    """
    results = {}
    for feat in target_features:
        if feat not in corr_matrix.columns:
            raise ValueError(f"Feature {feat} not found in correlation matrix.")

        # Sort by absolute correlation, ignore self (==1.0)
        corrs = corr_matrix[feat].drop(feat).abs().sort_values(ascending=False)
        top_feats = corrs.head(top_k).index
        results[feat] = [(f, corr_matrix.loc[feat, f]) for f in top_feats]

    return results

def log_high_correlation_feature_groups(
    data: Data,
    dataset_name: str,
    corr_threshold: float = 0.4,
    max_rows: int = 200000,
    top_k_label: int = 5,
) -> dict:
    """
    Print post-load feature diagnostics:
    1) feature-label correlations (top k)
    2) for those top features, list other highly correlated features

    Uses row sampling for very large datasets to keep runtime/memory bounded.
    Returns a dictionary of results for logging to JSON.
    """
    results = {
        "top_feature_label_corrs": {},
        "high_correlation_groups": {}
    }
    if not hasattr(data, "x") or data.x is None or not hasattr(data, "y") or data.y is None:
        return results
    if data.x.dim() != 2 or data.x.size(1) == 0:
        return results

    x = data.x.detach().cpu().float()
    y = data.y.detach().cpu().float().view(-1)
    n = x.size(0)
    d = x.size(1)
    if n < 3:
        return results

    # Subsample large datasets to avoid heavy diagnostics.
    if n > max_rows:
        idx = torch.randperm(n)[:max_rows]
        x = x[idx]
        y = y[idx]
        logger.info(f"[{dataset_name}] Correlation diagnostics sampled {max_rows}/{n} rows.")

    feature_names = getattr(data, "feature_names", None)
    if feature_names is None or len(feature_names) != d:
        feature_names = [f"feature_{i}" for i in range(d)]

    eps = 1e-12
    x_center = x - x.mean(dim=0, keepdim=True)
    x_std = x_center.std(dim=0, unbiased=False) + eps
    y_center = y - y.mean()
    y_std = y_center.std(unbiased=False) + eps

    # Pearson corr(feature_i, label)
    feature_label_corr = (x_center * y_center.unsqueeze(1)).mean(dim=0) / (x_std * y_std)
    abs_feature_label_corr = feature_label_corr.abs()
    
    # Take top k instead of thresholding
    top_k_actual = min(top_k_label, d)
    top_idx = torch.argsort(abs_feature_label_corr, descending=True)[:top_k_actual].tolist()

    top_preview = [
        f"{feature_names[i]}:{float(feature_label_corr[i]):.3f}"
        for i in top_idx
    ]
    logger.info(
        f"[{dataset_name}] Top {top_k_actual} feature-label correlations: {top_preview}"
    )
    
    for i in top_idx:
        results["top_feature_label_corrs"][feature_names[i]] = float(feature_label_corr[i])

    # Pearson corr matrix between input features.
    x_norm = x_center / x_std
    feat_corr = (x_norm.T @ x_norm) / x_norm.size(0)

    for i in top_idx:
        row = feat_corr[i].abs()
        neighbors = []
        for j in range(d):
            if j == i:
                continue
            if float(row[j]) > corr_threshold:
                neighbors.append((feature_names[j], float(feat_corr[i, j])))
        neighbors.sort(key=lambda t: abs(t[1]), reverse=True)
        
        if neighbors:
            results["high_correlation_groups"][feature_names[i]] = neighbors
            logger.info(
                f"[{dataset_name}] Highly correlated with '{feature_names[i]}' "
                f"(>|{corr_threshold:.2f}|): {neighbors}"
            )

    return results

from scipy.stats import pointbiserialr
from sklearn.feature_selection import f_classif

def analyze_feature_label_relation(
    data: Data,
    feature_names: Optional[list] = None,
    method: str = "auto",
    use_mutual_info: bool = False,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Analyze correlation/association between features (x) and labels (y).

    Args:
        data (Data): PyG Data object with `x` (features) and `y` (labels).
        feature_names (list, optional): Names of feature columns. If None, auto-generate.
        method (str): 'auto', 'pearson', 'spearman'. If 'auto', will choose based on label type.
        use_mutual_info (bool): For multi-class labels, use mutual information instead of ANOVA.
        top_k (int): Number of top features to return.

    Returns:
        pd.DataFrame: Ranked features with scores/correlations.
    """
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Data object has no features (x).")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Data object has no labels (y).")

    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy().ravel()

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    results = []

    # Case 1: Binary classification
    if np.unique(y).size == 2:
        for i, feat in enumerate(feature_names):
            try:
                r, _ = pointbiserialr(y, X[:, i])
                results.append((feat, abs(r), r))
            except Exception:
                results.append((feat, np.nan, np.nan))
        df = pd.DataFrame(results, columns=["feature", "abs_score", "score"]).sort_values(
            "abs_score", ascending=False
        )

    # Case 2: Multi-class classification
    elif np.unique(y).size > 2:
        if use_mutual_info:
            mi_scores = mutual_info_classif(X, y, discrete_features=False)
            df = pd.DataFrame({"feature": feature_names, "score": mi_scores})
            df["abs_score"] = df["score"]
            df = df.sort_values("score", ascending=False)
        else:
            f_scores, _ = f_classif(X, y)
            df = pd.DataFrame({"feature": feature_names, "score": f_scores})
            df["abs_score"] = df["score"]
            df = df.sort_values("score", ascending=False)

    # Case 3: Regression / continuous labels
    else:
        for i, feat in enumerate(feature_names):
            if method == "spearman":
                r = pd.Series(X[:, i]).corr(pd.Series(y), method="spearman")
            else:
                r = pd.Series(X[:, i]).corr(pd.Series(y), method="pearson")
            results.append((feat, abs(r), r))
        df = pd.DataFrame(results, columns=["feature", "abs_score", "score"]).sort_values(
            "abs_score", ascending=False
        )

    return df.head(top_k)


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(),
                        T.RandomNodeSplit('train_rest', num_val=500, num_test=500)])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(f'{root}/WIKICS', transform=None)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Coauthor(f'{root}/Coauthor', name, transform=None)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=None)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB', transform=None)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    data.feature_names = dataset.feature_names
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB', transform=None)
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(
    root: str,
    drop_features: Optional[list[str]] = None,
) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP', transform=None)
    data = dataset[0]
    feature_names = [f"feature_{i}" for i in range(data.x.size(1))]
    if drop_features:
        drop_set = set(drop_features)
        keep_indices = [i for i, feat in enumerate(feature_names) if feat not in drop_set]
        missing = sorted(drop_set - set(feature_names))
        if missing:
            logger.warning(
                f"[yelp] Requested drop_features not found and ignored: {missing}"
            )
        if len(keep_indices) == 0:
            raise ValueError(
                f"[yelp] drop_features removed all input features. Requested: {sorted(drop_set)}"
            )
        data.x = data.x[:, keep_indices]
        feature_names = [feature_names[i] for i in keep_indices]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    data.feature_names = feature_names
    return data, int(data.x.size(1)), dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr', transform=None)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2', transform=None)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes



def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train')
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes

def get_yelpchi(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = sio.loadmat(root, verify_compressed_data_integrity=False)
    num_features = dataset['features'].shape[1]
    num_classes = len(np.unique(dataset['label']))
    edge_index = torch.tensor([dataset['homo'].nonzero()[0], dataset['homo'].nonzero()[1]], dtype=torch.long)
    x = torch.tensor(pd.DataFrame.sparse.from_spmatrix(dataset['features']).values, dtype = torch.float)
    y = torch.tensor(dataset['label'][0], dtype=torch.long)
    train_mask, val_mask, test_mask = gen_masks(y, 0.5, 0.3, 20)
    data = Data(x=x, y=y, edge_index=edge_index, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return data, num_features, num_classes


def _resolve_graphland_dataset_path(root: str, name: str) -> str:
    root_abs = os.path.abspath(os.path.expanduser(root))
    root_parent = os.path.dirname(root_abs)
    root_grandparent = os.path.dirname(root_parent)
    candidates = [
        os.path.join(root, "graphland", "graphland", name),
        os.path.join(root, "graphland", name),
        os.path.join(root, name),
        os.path.join(root_parent, "graphland", "graphland", name),
        os.path.join(root_grandparent, "graphland", "graphland", name),
        os.path.join("data", "graphland", "graphland", name),
        os.path.join("data", "graphland", name),
        os.path.join("data", name),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "info.yaml")):
            return path
    raise FileNotFoundError(
        f"GraphLand dataset '{name}' not found. Expected info.yaml in one of: {candidates}"
    )


def _resolve_gadbench_dataset_path(root: str, name: str) -> str:
    root_abs = os.path.abspath(os.path.expanduser(root))
    root_parent = os.path.dirname(root_abs)
    root_grandparent = os.path.dirname(root_parent)
    candidates = [
        os.path.join(root, "datasets_gadbench", name),
        os.path.join(root, "datasets", "datasets_gadbench", name),
        os.path.join(root, name),
        os.path.join(root_parent, "datasets", "datasets_gadbench", name),
        os.path.join(root_grandparent, "datasets", "datasets_gadbench", name),
        os.path.join("data", "datasets", "datasets_gadbench", name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"GADBench dataset '{name}' not found. Expected one of: {candidates}"
    )


def _gadbench_dataset_exists(root: str, name: str) -> bool:
    try:
        _resolve_gadbench_dataset_path(root, name)
    except FileNotFoundError:
        return False
    return True


def _get_gadbench_split_col(config=None) -> int:
    """
    Match GADBench split convention:
    - columns 0..9: fully supervised trials
    - columns 10..19: semi-supervised trials
    """
    trial_id = 0
    semi_supervised = False
    if config is not None:
        eval_cfg = config.get("eval_params", {})
        pipe_cfg = config.get("pipeline_params", {})
        trial_id = int(
            eval_cfg.get("trial_id", pipe_cfg.get("gadbench_trial_id", 0))
        )
        semi_supervised = bool(
            eval_cfg.get(
                "semi_supervised",
                pipe_cfg.get("gadbench_semi_supervised", False)
            )
        )
    return trial_id + (10 if semi_supervised else 0)


def get_gadbench_dataset(
    root: str,
    name: str,
    config=None,
    drop_features: Optional[list[str]] = None,
) -> Tuple[Data, int, int]:
    """
    Load a preprocessed GADBench graph from DGL binary format and convert to PyG Data.
    Raw columns are named feature_0, feature_{D-1} (same convention as data.feature_names).
    """
    try:
        import dgl  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Loading GADBench datasets requires DGL. Install it first (e.g., pip install dgl)."
        ) from exc

    path = _resolve_gadbench_dataset_path(root, name)
    graph = dgl.load_graphs(path)[0][0]

    # Datasets in data/datasets/datasets_gadbench are typically already homogeneous, but
    # this keeps the loader robust if a heterograph sneaks in.
    if not getattr(graph, "is_homogeneous", False):
        graph = dgl.to_homogeneous(graph, ndata=["feature", "label"])

    x = graph.ndata["feature"].float()
    feature_names = [f"feature_{i}" for i in range(x.size(1))]
    if drop_features:
        drop_set = set(drop_features)
        keep_indices = [i for i, feat in enumerate(feature_names) if feat not in drop_set]
        dropped = [feat for feat in feature_names if feat in drop_set]
        missing = sorted(drop_set - set(dropped))
        if missing:
            logger.warning(
                f"[{name}] Requested drop_features not found and ignored: {missing}"
            )
        if len(keep_indices) == 0:
            raise ValueError(
                f"[{name}] drop_features removed all input features. "
                f"Requested: {sorted(drop_set)}"
            )
        x = x[:, keep_indices]
        feature_names = [feature_names[i] for i in keep_indices]
    y = graph.ndata["label"].long().view(-1)
    src, dst = graph.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)

    split_col = _get_gadbench_split_col(config)
    if "train_masks" in graph.ndata and "val_masks" in graph.ndata and "test_masks" in graph.ndata:
        split_col = min(split_col, graph.ndata["train_masks"].size(1) - 1)
        train_mask = graph.ndata["train_masks"][:, split_col].bool()
        val_mask = graph.ndata["val_masks"][:, split_col].bool()
        test_mask = graph.ndata["test_masks"][:, split_col].bool()
    elif "train_mask" in graph.ndata and "val_mask" in graph.ndata and "test_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"].bool()
        val_mask = graph.ndata["val_mask"].bool()
        test_mask = graph.ndata["test_mask"].bool()
    else:
        train_mask, val_mask, test_mask = gen_masks(y, 0.4, 0.2, 1)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.feature_names = feature_names

    num_features = int(x.size(1))
    num_classes = int(y.max().item() + 1)
    return data, num_features, num_classes


def _one_hot_encode(
    categorical_features: np.ndarray,
    categorical_feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    if categorical_features.shape[1] == 0:
        return categorical_features, []
    try:
        encoder = OneHotEncoder(drop="if_binary", sparse_output=False, dtype=np.float32)
    except TypeError:
        encoder = OneHotEncoder(drop="if_binary", sparse=False, dtype=np.float32)
    if categorical_feature_names is not None and len(categorical_feature_names) == categorical_features.shape[1]:
        cat_df = pd.DataFrame(categorical_features, columns=categorical_feature_names)
        encoded = encoder.fit_transform(cat_df)
    else:
        encoded = encoder.fit_transform(categorical_features)
    if hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out().tolist()
    else:
        feature_names = [f"cat_{i}" for i in range(encoded.shape[1])]
    return encoded, feature_names


def get_graphland_dataset(
    root: str,
    name: str,
    split: str = "RL",
    add_self_loops: bool = False,
    to_undirected_graph: bool = True,
    drop_features: list[str] | None = None,
) -> Tuple[Data, int, int]:
    dataset_path = _resolve_graphland_dataset_path(root, name)

    with open(os.path.join(dataset_path, "info.yaml"), "r") as file:
        info = yaml.safe_load(file)

    fraction_features_names = info.get("fraction_features_names", [])
    numerical_features_names = [
        feat for feat in info.get("numerical_features_names", []) if feat not in set(fraction_features_names)
    ]
    categorical_features_names = info.get("categorical_features_names", [])

    features_df = pd.read_csv(os.path.join(dataset_path, "features.csv"), index_col=0)
    numerical_features = features_df[numerical_features_names].values.astype(np.float32)
    fraction_features = features_df[fraction_features_names].values.astype(np.float32)
    categorical_features = features_df[categorical_features_names].values.astype(np.float32)
    categorical_features, categorical_feature_names = _one_hot_encode(
        categorical_features,
        categorical_feature_names=categorical_features_names,
    )

    features = np.concatenate([numerical_features, fraction_features, categorical_features], axis=1)
    feature_names = (
        numerical_features_names
        + fraction_features_names
        + (categorical_feature_names if categorical_feature_names else [])
    )
    if drop_features:
        drop_set = set(drop_features)
        keep_indices = [i for i, feat in enumerate(feature_names) if feat not in drop_set]
        dropped = [feat for feat in feature_names if feat in drop_set]
        missing = sorted(drop_set - set(dropped))
        if missing:
            logger.warning(
                f"[{name}] Requested drop_features not found and ignored: {missing}"
            )
        if len(keep_indices) == 0:
            raise ValueError(
                f"[{name}] drop_features removed all input features. "
                f"Requested: {sorted(drop_set)}"
            )
        features = features[:, keep_indices]
        feature_names = [feature_names[i] for i in keep_indices]
        logger.info(
            f"[{name}] Dropped {len(dropped)} feature(s): {dropped}. "
            f"Remaining features: {len(feature_names)}"
        )

    targets = pd.read_csv(os.path.join(dataset_path, "targets.csv"), index_col=0).values.squeeze(1).astype(np.float32)
    task_type = str(info.get("task", "classification")).lower()
    labeled_mask = ~np.isnan(targets)
    targets_filled = targets.copy()
    targets_filled[~labeled_mask] = 0

    edges_df = pd.read_csv(os.path.join(dataset_path, "edgelist.csv"))
    edges = edges_df.values[:, :2].astype(np.int64)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    if to_undirected_graph:
        edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

    split_masks_df = pd.read_csv(os.path.join(dataset_path, f"split_masks_{split}.csv"), index_col=0)
    train_mask_orig = split_masks_df["train"].values.astype(bool)
    val_mask_orig = split_masks_df["val"].values.astype(bool)
    test_mask_orig = split_masks_df["test"].values.astype(bool)

    train_mask = torch.tensor(train_mask_orig & labeled_mask)
    val_mask = torch.tensor(val_mask_orig & labeled_mask)
    test_mask = torch.tensor(test_mask_orig & labeled_mask)

    x = torch.tensor(features, dtype=torch.float32)
    if task_type == "regression":
        y = torch.tensor(targets_filled, dtype=torch.float32)
        num_classes = 1
    else:
        y = torch.tensor(targets_filled.astype(np.int64), dtype=torch.long)
        num_classes = int(np.max(targets[labeled_mask]) + 1) if labeled_mask.any() else 0

    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(x.size(1))]
    data.task_type = task_type
    return data, x.size(1), num_classes


def skew_pokec_data(data: Data, drop_quadrant: str, seed: int = 42) -> Data:
    """
    Skew the Pokec dataset by removing 50% of nodes from the chosen quadrant.
    drop_quadrant ∈ {'F0','M0','F1','M1'}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gender_idx = data.feature_names.index('gender')
    genders = data.x[:, gender_idx].cpu().numpy().astype(int)
    labels = data.y.cpu().numpy().astype(int)

    # quadrant mask
    if drop_quadrant[0] == 'F':
        g_mask = genders == 0
    else:
        g_mask = genders == 1
    l_mask = labels == int(drop_quadrant[1])
    q_indices = np.where(g_mask & l_mask)[0]

    # randomly drop 50% of this quadrant
    drop_size = len(q_indices) // 2
    drop_nodes = np.random.choice(q_indices, size=drop_size, replace=False)
    keep_nodes = np.setdiff1d(np.arange(data.num_nodes), drop_nodes)

    # make subgraph - convert numpy array to torch tensor
    keep_nodes_tensor = torch.from_numpy(keep_nodes)
    sub_edge_index, _ = subgraph(
        keep_nodes_tensor, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    new_x = data.x[keep_nodes]
    new_y = data.y[keep_nodes]

    new_data = Data(x=new_x, y=new_y, edge_index=sub_edge_index)
    new_data.feature_names = data.feature_names

    # re-generate masks with same proportions
    train_mask, val_mask, test_mask = gen_masks(new_y, 0.5, 0.3, 1)
    new_data.train_mask = train_mask
    new_data.val_mask = val_mask
    new_data.test_mask = test_mask

    print(f"Skewed dataset {drop_quadrant}: dropped {drop_size}, remaining {new_data.num_nodes}")
    return new_data

def get_pokec(root: str, name: str='pokec_n', config=None, filter_neg_labels: bool = True, drop_features: list[str]|None = None) -> Tuple[Data, int, int]:
    """
    Load the Pokec dataset and return a torch_geometric Data object with
    x, y, edge_index, and boolean train/val/test masks, consistent with other loaders.
    Expects files "{dataset}.csv" and "{dataset}_relationship.txt" under root.
    
    Args:
        root: Path to the dataset directory
        name: Dataset variant ('pokec_z', 'pokec_n', etc.)
        filter_neg_labels: If True, removes datapoints where labels are -1. If False, keeps all datapoints.
    """
    # Map common aliases to file stems used by public Pokec preprocessors
    if name == 'pokec_z':
        dataset = 'region_job'
    elif name == 'pokec_n':
        dataset = 'region_job_2'
    else:
        dataset = 'region_job_2'

    path = root
    sens_attr = ''
    predict_attr = 'I_am_working_in_field'

    # Load tabular features/labels
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    if 'user_id' in header:
        header.remove('user_id')
    # target column should not be included as feature
    if predict_attr in header:
        header.remove(predict_attr)

    # for training without the sensitive attribute
    # if sens_attr in header:
    #     header.remove(sens_attr)
    # Apply filtering based on the flag
    if filter_neg_labels:
        # Filter out rows where the target label is -1
        valid_label_mask = idx_features_labels[predict_attr] != -1
        idx_features_labels_filtered = idx_features_labels[valid_label_mask].reset_index(drop=True)
    else:
        # Keep all datapoints including those with -1 labels
        idx_features_labels_filtered = idx_features_labels

    # Preserve feature name ordering for downstream reference
    feature_names = list(header)
    if drop_features is not None:
        feature_names = [f for f in feature_names if f not in drop_features]
    
    print("After dropping the mentioned features, the following features are being used for pretraining: ")
    print(feature_names)

    # Impute missing age values (where AGE is 0.0) by sampling from existing distribution
    if 'AGE' in idx_features_labels_filtered.columns:
        age_col = idx_features_labels_filtered['AGE']
        # Count missing values (0.0 or NaN)
        missing_age_mask = (age_col == 0.0) | age_col.isna()
        missing_count = missing_age_mask.sum()
        total_count = len(age_col)

        if missing_count > 0:
            # Sample with replacement from valid (non-missing, non-zero) ages to preserve distribution
            valid_ages = age_col[(age_col != 0.0) & ~age_col.isna()].values
            if valid_ages.size == 0:
                # Fallback to no-op if no valid ages (should not happen in practice)
                print("Age imputation: No valid ages to sample from; skipping.")
            else:
                sampled = np.random.choice(valid_ages, size=int(missing_count), replace=True)
                idx_features_labels_filtered.loc[missing_age_mask, 'AGE'] = sampled
                print(f"Age imputation: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%) missing values imputed by sampling from empirical age distribution")
        else:
            print("Age imputation: No missing age values found")
    
    features_np = idx_features_labels_filtered[feature_names].to_numpy(dtype=np.float32)
    x = torch.tensor(features_np, dtype=torch.float32)

    labels_np = idx_features_labels_filtered[predict_attr].to_numpy()
    labels_np[labels_np > 1] = 1
    
    # Shift labels to make them non-negative if there are negative labels
    min_label = labels_np.min()
    if min_label < 0:
        labels_np = labels_np - min_label
        print(f"Shifted labels by {-min_label} to make non-negative. New range: {labels_np.min()} to {labels_np.max()}")
    
    y = torch.tensor(labels_np, dtype=torch.long)

    # Build edge_index from relationship file
    idx = idx_features_labels_filtered['user_id'].to_numpy(dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, f"{dataset}_relationship.txt"), dtype=int)
    if edges_unordered.ndim == 1 and edges_unordered.size == 0:
        edges_unordered = np.empty((0, 2), dtype=int)
    if edges_unordered.ndim == 1 and edges_unordered.size == 2:
        edges_unordered = edges_unordered.reshape(1, 2)
    # Keep only edges whose both endpoints exist in user_id list (after filtering)
    valid_src = np.isin(edges_unordered[:, 0], idx)
    valid_dst = np.isin(edges_unordered[:, 1], idx)
    valid_mask = valid_src & valid_dst
    edges_unordered = edges_unordered[valid_mask]
    # Map raw ids to contiguous indices
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(-1, 2)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # Create masks similar to YelpChi processing (50% train, 30% val)
    train_mask, val_mask, test_mask = gen_masks(y, 0.5, 0.3, 1)

    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    # Expose feature names for name-based selection in visualization/corruption
    try:
        data.feature_names = feature_names  # type: ignore[attr-defined]
    except Exception:
        pass

    vis_root = os.path.join(config['management']['output_folder_dir'], "visualization_plots")
    vis_dir = os.path.join(vis_root, f"{config['eval_params']['dataset']}_{config['pipeline_params']['model_name']}")
    os.makedirs(vis_dir, exist_ok=True)

    # Create a DataFrame for analysis including demographic features and labels
    plot_df = idx_features_labels_filtered.copy()
    plot_df['label'] = labels_np  # Use the processed labels
    
    # # Create figure with subplots - expanded to 3x2 layout
    # fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    # fig.suptitle(f'Label Distribution Analysis for {name}', fontsize=16, fontweight='bold')
    
    # # 1. Age distribution by label
    # if 'AGE' in plot_df.columns:
    #     ax1 = axes[0, 0]
    #     for label_val in sorted(plot_df['label'].unique()):
    #         age_data = plot_df[plot_df['label'] == label_val]['AGE'].dropna()
    #         ax1.hist(age_data, bins=30, alpha=0.6, label=f'Label {label_val}', density=False)
    #     ax1.set_xlabel('Age')
    #     ax1.set_ylabel('Count')
    #     ax1.set_title('Age Distribution by Label')
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
    
    # # 2. Gender distribution by label
    # if 'gender' in plot_df.columns:
    #     ax2 = axes[0, 1]
    #     gender_label_counts = pd.crosstab(plot_df['gender'], plot_df['label'], normalize=False)
    #     gender_label_counts.plot(kind='bar', ax=ax2, rot=0)
    #     ax2.set_xlabel('Gender')
    #     ax2.set_ylabel('Count')
    #     ax2.set_title('Gender Distribution by Label')
    #     ax2.legend(title='Label')
    
    # # 3. Region distribution by label
    # if 'region' in plot_df.columns:
    #     ax3 = axes[1, 0]
    #     region_label_counts = pd.crosstab(plot_df['region'], plot_df['label'])
    #     # Show top 10 regions by total count to avoid overcrowding
    #     top_regions = region_label_counts.sum(axis=1).nlargest(10).index
    #     region_subset = region_label_counts.loc[top_regions]
    #     region_subset.plot(kind='bar', ax=ax3, rot=45)
    #     ax3.set_xlabel('Region (Top 10)')
    #     ax3.set_ylabel('Count')
    #     ax3.set_title('Region Distribution by Label (Top 10 Regions)')
    #     ax3.legend(title='Label')
    
    # # 4. English language proficiency by label
    # if 'anglicky' in plot_df.columns:
    #     ax4 = axes[1, 1]
    #     anglicky_label_counts = pd.crosstab(plot_df['anglicky'], plot_df['label'], normalize=False)
    #     anglicky_label_counts.plot(kind='bar', ax=ax4, rot=0)
    #     ax4.set_xlabel('English Proficiency')
    #     ax4.set_ylabel('Count')
    #     ax4.set_title('English Proficiency Distribution by Label')
    #     ax4.legend(title='Label')
    
    # # 5. Marital status by label
    # if 'zenaty (vydata)' in plot_df.columns:
    #     ax5 = axes[2, 0]
    #     marital_label_counts = pd.crosstab(plot_df['zenaty (vydata)'], plot_df['label'], normalize=False)
    #     marital_label_counts.plot(kind='bar', ax=ax5, rot=0)
    #     ax5.set_xlabel('Marital Status')
    #     ax5.set_ylabel('Count')
    #     ax5.set_title('Marital Status Distribution by Label')
    #     ax5.legend(title='Label')
    
    # # 6. Label distribution summary
    # ax6 = axes[2, 1]
    # label_counts = plot_df['label'].value_counts().sort_index()
    # colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(label_counts)]
    # bars = ax6.bar(label_counts.index, label_counts.values, color=colors)
    # ax6.set_xlabel('Label')
    # ax6.set_ylabel('Count')
    # ax6.set_title('Overall Label Distribution')
    # ax6.set_xticks(label_counts.index)
    
    # # Add value labels on bars
    # for bar, count in zip(bars, label_counts.values):
    #     ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(label_counts.values)*0.01,
    #             str(count), ha='center', va='bottom', fontweight='bold')
    
    # plt.tight_layout()
    
    # # Save the plot instead of showing it
    # plot_filename = os.path.join(vis_dir, f"{name}_label_distribution_analysis.png")
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # plt.close()  # Close the figure to free memory
    
    # print(f"Saved visualization plot to: {plot_filename}")
    
    # # Print summary statistics
    # print("\n" + "="*60)
    # print("DEMOGRAPHIC ANALYSIS SUMMARY")
    # print("="*60)
    
    # print(f"\nDataset: {name}")
    # print(f"Total samples: {len(plot_df)}")
    # print(f"Label distribution:")
    # for label_val, count in label_counts.items():
    #     percentage = count / len(plot_df) * 100
    #     print(f"  Label {label_val}: {count} ({percentage:.1f}%)")
    
    # # Age statistics by label
    # if 'AGE' in plot_df.columns:
    #     print(f"\nAge statistics by label:")
    #     age_stats = plot_df.groupby('label')['AGE'].agg(['count', 'mean', 'std', 'min', 'max'])
    #     print(age_stats.round(2))
    
    # # Gender distribution by label
    # if 'gender' in plot_df.columns:
    #     print(f"\nGender distribution by label:")
    #     gender_crosstab = pd.crosstab(plot_df['gender'], plot_df['label'], margins=True)
    #     print(gender_crosstab)
    
    # # Region statistics
    # if 'region' in plot_df.columns:
    #     print(f"\nRegion statistics:")
    #     print(f"  Total unique regions: {plot_df['region'].nunique()}")
    #     region_label_dist = pd.crosstab(plot_df['region'], plot_df['label'])
    #     print(f"  Top 5 regions by total count:")
    #     top_5_regions = region_label_dist.sum(axis=1).nlargest(5)
    #     for region, count in top_5_regions.items():
    #         print(f"    Region {region}: {count} samples")
    
    # # English proficiency distribution by label
    # if 'anglicky' in plot_df.columns:
    #     print(f"\nEnglish proficiency distribution by label:")
    #     anglicky_crosstab = pd.crosstab(plot_df['anglicky'], plot_df['label'], margins=True)
    #     print(anglicky_crosstab)
    
    # # Marital status distribution by label
    # if 'zenaty (vydata)' in plot_df.columns:
    #     print(f"\nMarital status distribution by label:")
    #     marital_crosstab = pd.crosstab(plot_df['zenaty (vydata)'], plot_df['label'], margins=True)
    #     print(marital_crosstab)
    
    # print("="*60)

    # # Correlation with labels included
    # corr_df = analyze_feature_correlation(
    #     data,
    #     feature_names=data.feature_names,
    #     method="spearman",
    #     include_labels=True
    # )    
    # print(corr_df.head())

    # results = get_top_correlated_features(corr_df, ["AGE", "region", "gender", "anglicky", "zenaty (vydata)", "label"], top_k=5)

    # for feat, correlated in results.items():
    #     print(f"\nTop correlated with {feat}:")
    #     for other_feat, val in correlated:
    #         print(f"  {other_feat}: {val:.4f}")

    # top_corr = corr_df["label"].drop("label").abs().sort_values(ascending=False).head(10)
    # print("Top features correlated with label:")
    # print(top_corr)

    # # Top features correlated with binary label
    # df_label_corr = analyze_feature_label_relation(
    #     data, 
    #     feature_names=data.feature_names, 
    #     top_k=10
    # )

    # print(df_label_corr)

    return data, x.size(1), int(y.max().item() + 1)

def get_credit(root: str,drop_features: list[str] | None = None) -> Tuple[Data, int, int]:
    """
    Load and preprocess the Credit dataset (binary classification: NoDefaultNextMonth).
    Emulates the preprocessing of the original load_credit() but returns a PyG Data object.
    """
    import random
    root = os.path.expanduser(root)
    path = os.path.join(root, "credit")
    dataset = "credit"

    csv_path = os.path.join(path, f"{dataset}.csv")
    edges_path = os.path.join(path, f"{dataset}_edges.txt")

    print(f"edges path: {edges_path}")

    df = pd.read_csv(csv_path)
    predict_attr = "NoDefaultNextMonth"
    sens_attr = "Age"

    header = list(df.columns)
    header.remove(predict_attr)
    if drop_features is not None:
        header = [f for f in header if f not in drop_features]

    print("[CREDIT] Using features:")
    print(header)
    # if "Single" in header:
    #     header.remove("Single")  # matches original code
    # WHYYY

    # load or construct edges
    if os.path.exists(edges_path):
        edges_unordered = np.genfromtxt(edges_path).astype(int)
    else:
        raise FileNotFoundError(f"Missing edge file: {edges_path}")

    # sparse features
    features = sp.csr_matrix(df[header], dtype=np.float32)
    labels = df[predict_attr].values.astype(int)

    # build symmetric adjacency
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])  # add self-loops

    # convert to torch
    x = torch.FloatTensor(np.array(features.todense()))
    y = torch.LongTensor(labels)

    # stratified split (same logic)
    random.seed(20)
    label_idx_0 = np.where(y == 0)[0]
    label_idx_1 = np.where(y == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:int(0.5 * len(label_idx_0))],
        label_idx_1[:int(0.5 * len(label_idx_1))]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    # build edge_index from adjacency
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # build masks
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # sensitive attribute
    sens = torch.FloatTensor(df[sens_attr].values.astype(int))
    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.sens = sens
    data.feature_names = header

    print(f"[CREDIT] Loaded: {x.size(0)} nodes, {edge_index.size(1)} edges, "
          f"{x.size(1)} features, classes={y.unique().tolist()}")
    return data, x.size(1), int(y.max().item() + 1)


def get_income(root: str,drop_features: list[str] | None = None) -> Tuple[Data, int, int]:
    """
    Load and preprocess the Income dataset (binary classification: income).
    Emulates the preprocessing of the original load_income() but returns a PyG Data object.
    """
    import random
    root = os.path.expanduser(root)
    path = os.path.join(root, "income")
    dataset = "income"

    csv_path = os.path.join(path, f"{dataset}.csv")
    edges_path = os.path.join(path, f"{dataset}_edges.txt")

    df = pd.read_csv(csv_path)
    predict_attr = "income"
    sens_attr = "race"

    header = list(df.columns)
    header.remove(predict_attr)

    if drop_features is not None:
        header = [f for f in header if f not in drop_features]

    print("[INCOME] Using features:")
    print(header)

    # load or construct edges
    if os.path.exists(edges_path):
        edges_unordered = np.genfromtxt(edges_path).astype(int)
    else:
        raise FileNotFoundError(f"Missing edge file: {edges_path}")

    # sparse features
    features = sp.csr_matrix(df[header], dtype=np.float32)
    labels = df[predict_attr].values.astype(int)

    # build symmetric adjacency
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])  # add self-loops

    # convert to torch
    x = torch.FloatTensor(np.array(features.todense()))
    y = torch.LongTensor(labels)

    # stratified split (same as original)
    random.seed(20)
    label_idx_0 = np.where(y == 0)[0]
    label_idx_1 = np.where(y == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:int(0.5 * len(label_idx_0))],
        label_idx_1[:int(0.5 * len(label_idx_1))]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    # build edge_index from adjacency
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # build masks
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # sensitive attribute
    sens = torch.FloatTensor(df[sens_attr].values.astype(int))
    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.sens = sens
    data.feature_names = header

    print(f"[INCOME] Loaded: {x.size(0)} nodes, {edge_index.size(1)} edges, "
          f"{x.size(1)} features, classes={y.unique().tolist()}")
    return data, x.size(1), int(y.max().item() + 1)


def get_bail(root: str, drop_features: list[str] | None = None) -> Tuple[Data, int, int]:
    """
    Load and preprocess the Bail dataset (binary classification: RECID).
    Emulates the preprocessing of the original load_bail() but returns a PyG Data object.
    """
    import random
    root = os.path.expanduser(root)
    path = os.path.join(root, "bail")
    dataset = "bail"

    csv_path = os.path.join(path, f"{dataset}.csv")
    edges_path = os.path.join(path, f"{dataset}_edges.txt")

    df = pd.read_csv(csv_path)
    predict_attr = "RECID"
    sens_attr = "WHITE"

    header = list(df.columns)
    header.remove(predict_attr)

    if drop_features is not None:
        header = [f for f in header if f not in drop_features]

    print("[BAIL] Using features:")
    print(header)

    # load or construct edges
    if os.path.exists(edges_path):
        edges_unordered = np.genfromtxt(edges_path).astype(int)
    else:
        raise FileNotFoundError(f"Missing edge file: {edges_path}")

    # sparse features
    features = sp.csr_matrix(df[header], dtype=np.float32)
    labels = df[predict_attr].values.astype(int)

    # build symmetric adjacency
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])  # add self-loops

    # convert to torch
    x = torch.FloatTensor(np.array(features.todense()))
    y = torch.LongTensor(labels)

    # stratified split (same as original)
    random.seed(20)
    label_idx_0 = np.where(y == 0)[0]
    label_idx_1 = np.where(y == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:int(0.5 * len(label_idx_0))],
        label_idx_1[:int(0.5 * len(label_idx_1))]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    # build edge_index from adjacency
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # build masks
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # sensitive attribute
    sens = torch.FloatTensor(df[sens_attr].values.astype(int))
    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.sens = sens
    data.feature_names = header

    print(f"[BAIL] Loaded: {x.size(0)} nodes, {edge_index.size(1)} edges, "
          f"{x.size(1)} features, classes={y.unique().tolist()}")
    return data, x.size(1), int(y.max().item() + 1)


def get_data(root: str, name: str, config=None) -> Tuple[Data, int, int]:
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        data, nf, nc = get_planetoid(root, name)
    elif name.lower() in ['coauthorcs', 'coauthorphysics']:
        data, nf, nc = get_coauthor(root, name[8:])
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        data, nf, nc = get_amazon(root, name[6:])
    elif name.lower() == 'wikics':
        data, nf, nc = get_wikics(root)
    elif name.lower() in ['cluster', 'pattern']:
        data, nf, nc = get_sbm(root, name)
    elif name.lower() == 'reddit2':
        data, nf, nc = get_reddit(root)
    elif name.lower() == 'flickr':
        data, nf, nc = get_flickr(root)
    elif name.lower() == 'yelp':
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)
        # Prefer local GADBench Yelp if present, otherwise fall back to PyG Yelp.
        if _gadbench_dataset_exists(root, "yelp"):
            data, nf, nc = get_gadbench_dataset(
                root, "yelp", config=config, drop_features=drop_features
            )
        else:
            data, nf, nc = get_yelp(root, drop_features=drop_features)
    elif name.lower() in ['tsocial', 't-social']:
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)
        data, nf, nc = get_gadbench_dataset(
            root, "tsocial", config=config, drop_features=drop_features
        )
    elif name.lower() in ['tfinance', 't-finance']:
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)
        data, nf, nc = get_gadbench_dataset(
            root, "tfinance", config=config, drop_features=drop_features
        )
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        data, nf, nc = get_arxiv(root)
    elif name.lower() in ['ogbn-products', 'products']:
        data, nf, nc = get_products(root)
    elif name.lower() in ['yelpchi']:
        data, nf, nc = get_yelpchi('./datasets/YelpChi.mat', 'yelpchi')
    elif name.lower() in ['pokec']:
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)

        data, nf, nc = get_pokec(
            root + '/pokec',
            name,
            config,
            drop_features=drop_features
        )
        #data, nf, nc = get_pokec(root+'/pokec', name, config,drop_features=None)
        # data = skew_pokec_data(data,'F1')
    elif name.lower() == 'credit':
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)

        data, nf, nc = get_credit(root,drop_features=drop_features)
    elif name.lower() == 'income':
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)

        data, nf, nc = get_income(root, drop_features=drop_features)
    elif name.lower() == 'bail':
        drop_features = None
        if config is not None:
            drop_features = config["pipeline_params"].get("drop_features", None)

        data, nf, nc = get_bail(root, drop_features=drop_features)
    elif name.lower() in ['web-fraud', 'web_fraud', 'webfraud']:
        split = "RL"
        drop_features = None
        if config is not None:
            split = config.get("eval_params", {}).get("split", split)
            split = config.get("pipeline_params", {}).get("graphland_split", split)
            drop_features = config["pipeline_params"].get("drop_features", None)
        data, nf, nc = get_graphland_dataset(root, "web-fraud", split=split, drop_features=drop_features)
    elif name.lower() in ['twitch-views', 'twitch_views', 'twitchviews']:
        split = "RH"
        drop_features = None
        if config is not None:
            split = config.get("eval_params", {}).get("split", split)
            split = config.get("pipeline_params", {}).get("graphland_split", split)
            drop_features = config["pipeline_params"].get("drop_features", None)
        data, nf, nc = get_graphland_dataset(root, "twitch-views", split=split, drop_features=drop_features)
    elif name.lower() in ['artnet-views', 'artnet_views', 'artnetviews']:
        split = "RH"
        drop_features = None
        if config is not None:
            split = config.get("eval_params", {}).get("split", split)
            split = config.get("pipeline_params", {}).get("graphland_split", split)
            drop_features = config["pipeline_params"].get("drop_features", None)
        data, nf, nc = get_graphland_dataset(root, "artnet-views", split=split, drop_features=drop_features)
    else:
        raise NotImplementedError

    # Log feature names (dataset-agnostic). Do not mutate the data object.
    try:
        feature_names = getattr(data, 'feature_names', None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(nf)]
        preview = feature_names[:30]
        more = ' ...' if len(feature_names) > 30 else ''
        logger.info(f"[{name}] Feature names ({len(feature_names)}): {preview}{more}")
        

    except Exception:
        pass

    edge_index = data.edge_index
    deg = torch.bincount(edge_index[0], minlength=data.num_nodes)
    data.degree = deg
    # Post-load diagnostics: feature-label and feature-feature correlations.
    try:
        data.corr_diagnostics = log_high_correlation_feature_groups(data, name, corr_threshold=0.4, top_k_label=5)
    except Exception as exc:
        logger.warning(f"[{name}] Failed correlation diagnostics: {exc}")
        data.corr_diagnostics = {}

    return data, nf, nc


def to_inductive(data):
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    i = 1
    while hasattr(data, f'x{i}'):
        data[f'x{i}'] = data[f'x{i}'][mask]
        i += 1
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def preprocess_data(model_config, data):
    loop, normalize = model_config['loop'], model_config['normalize']
    if loop:
        t = time.perf_counter()
        logger.info('Adding self-loops... ')
        data.adj_t = data.adj_t.set_diag()
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    if normalize:
        t = time.perf_counter()
        data.adj_t = gcn_norm(data.adj_t)
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')


def prepare_dataset(model_config, data, remove_edge_index=True):
    print("hi debug 6")
    train_data = to_inductive(data)
    print("hi debug 7") 
    train_data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(train_data.to(get_device()))
    print("hi debug 8")
    data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(data.to(get_device()))
    print("hi debug 9")
    preprocess_data(model_config, train_data)
    print("hi debug 10")
    preprocess_data(model_config, data)
    return train_data, data

"""
Dataset-sensitive feature metadata used by editing suite runs.

Keep in sync with ``editing_pipelines/run_editing_suite.sh``:
  - ``SENSITIVE_FEATURE_MAP`` (bash associative array)
  - ``FIXED_VALUE_MAP`` (optional comma-separated anchors per dataset)

Datasets omitted here (e.g. ``credit`` with a commented entry in the shell) fall back to
``run_edit.build_config`` when the ablation script uses legacy mapping.
"""

from typing import Dict, List, Optional

# Mirrors: declare -A SENSITIVE_FEATURE_MAP=( ... )
SENSITIVE_FEATURE_BY_DATASET: Dict[str, str] = {
    "bail": "WHITE",
    "income": "fnlwgt",
    "pokec": "AGE",
    "yelp": "feature_5",
    "tfinance": "feature_8",
    "artnet-views": "feature_20_fraction",
    "twitch-views": "affiliate_status_1.0",
}

# Mirrors: declare -A FIXED_VALUE_MAP=( ... ) — missing keys => no fixed anchors (None).
FIXED_SENSITIVE_VALUES_BY_DATASET: Dict[str, Optional[List]] = {
    "bail": [0, 1],
    "pokec": [15, 25, 35, 45, 55, 65],
    "yelp": [0.39845687, 0.99998516],
}


def apply_suite_sensitive_feature_config(config: dict) -> dict:
    """
    Patch ``config['pipeline_params']`` for ``sensitive_feature`` and
    ``fixed_sensitive_values`` when the dataset appears in the suite maps.
    Returns the same config for chaining.
    """
    dataset_key = str(config.get("eval_params", {}).get("dataset", "")).lower()
    pp = config.setdefault("pipeline_params", {})

    if dataset_key in SENSITIVE_FEATURE_BY_DATASET:
        pp["sensitive_feature"] = SENSITIVE_FEATURE_BY_DATASET[dataset_key]

    if dataset_key in SENSITIVE_FEATURE_BY_DATASET:
        pp["fixed_sensitive_values"] = FIXED_SENSITIVE_VALUES_BY_DATASET.get(dataset_key)

    return config

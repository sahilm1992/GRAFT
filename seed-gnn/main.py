import sys
import os
import json
import datetime
from zoneinfo import ZoneInfo

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(current_dir)
sys.path.append(base_dir)
os.chdir(base_dir)

import main_utils as main_utils
from pipelines.seed_gnn.eval_edit_gnn import eval_edit_gnn
from pipelines.seed_gnn.pretrain_gnn import pretrain_gnn
from pipelines.seed_gnn.run_modcirc import run_neuron_discovery
from constants import NODE_CLASSIFICATION_DATASETS, NODE_REGRESSION_DATASETS, SEED

# main_utils.set_seeds_all(SEED) # Removed; now set after config is loaded

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = main_utils.parse_args()
config = main_utils.register_args_and_configs(args) # config["management"]["seed"] is set here
logger = main_utils.set_logger(args.output_folder_dir, args)
main_utils.set_seeds_all(config["management"]["seed"]) # Set seeds with the config's seed after it's loaded


use_only_edit_examples = False

logger.info(f"Experiment {config['management']['exp_desc']} (SEED={config['management']['seed']}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))
# ================= FEATURE-ABLATION CONFIG INJECTION =================

pp = config.setdefault("pipeline_params", {})

# ---- Defaults (safe for old configs) ----
pp.setdefault("feature_variant", "full_features")
pp.setdefault("drop_features", [])
pp.setdefault("use_feature_ablated_ckpts", False)

# ---- Override from command line arguments if provided ----
if args.use_feature_ablation:
    pp["feature_variant"] = args.feature_variant
    pp["drop_features"] = args.drop_features
    pp["use_feature_ablated_ckpts"] = True
elif pp.get("use_feature_ablated_ckpts"):
    # If not explicitly set via CLI but present in config (rare but possible)
    pass

feature_variant = pp["feature_variant"]
use_feature_ablated = pp["use_feature_ablated_ckpts"]

# -------- CHECKPOINT DIRECTORY --------
ckpt_base = config["management"]["pretrain_output_dir"]

if use_feature_ablated:
    ckpt_base = os.path.join(
        os.path.dirname(ckpt_base),
        "edit_ckpts_feature_ablated"
    )

dataset = config["eval_params"]["dataset"]
ckpt_dir = os.path.join(ckpt_base, dataset, feature_variant)
os.makedirs(ckpt_dir, exist_ok=True)

config["management"]["pretrain_output_dir"] = ckpt_dir

# -------- RESULTS DIRECTORY --------
results_base = config["management"]["output_folder_dir"]
results_dir = os.path.join(results_base, feature_variant)
os.makedirs(results_dir, exist_ok=True)

config["management"]["output_folder_dir"] = results_dir

# -------- Sanity logging --------
logger.info(f"[FEATURE VARIANT] {feature_variant}")
logger.info(f"[DROP FEATURES] {pp['drop_features']}")
logger.info(f"[FEATURE-ABLATED CKPTS] {use_feature_ablated}")
logger.info(f"[CKPT DIR] {ckpt_dir}")
logger.info(f"[RESULTS DIR] {results_dir}")


dataset_name = config['eval_params']['dataset']
task_name = config['management']['task']
is_classification_dataset = dataset_name in NODE_CLASSIFICATION_DATASETS
is_regression_dataset = dataset_name in NODE_REGRESSION_DATASETS

if task_name == 'pretrain':
    if is_classification_dataset or is_regression_dataset:
        pretrain_gnn(config)
    else:
        logger.error(f"Invalid dataset for pretrain: {dataset_name}.")
        raise ValueError
elif task_name == 'edit':
    if not (is_classification_dataset or is_regression_dataset):
        logger.error(f"Editing currently supports node classification/regression datasets only. Got: {dataset_name}.")
        raise ValueError
    config['management']['debug'] = bool(config['management'].get('debug', False))
    if use_only_edit_examples:
        config['pipeline_params']['beta']=0
    raw_results, processed_results = eval_edit_gnn(
        config,
        debug=config['management']['debug'],
    )
    main_utils.register_result(raw_results, config)
    config['eval_results'] = processed_results
elif task_name == 'modcirc':
    if not is_classification_dataset:
        logger.error(f"modcirc currently supports node-classification datasets only. Got: {dataset_name}.")
        raise ValueError
    run_neuron_discovery(config)
else:
    logger.error(f"Invalid args.task input: {task_name}.")
    raise ValueError

end_time = datetime.datetime.now(ct_timezone)
main_utils.register_exp_time(start_time, end_time, config)
main_utils.register_output_config(config)
logger.info(f"Experiment {config['management']['exp_desc']} ended at {end_time}. Duration: {config['management']['exp_duration']}")

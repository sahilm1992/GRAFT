import time
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import from the seed-gnn directory
import sys
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')
from edit_gnn.utils import grab_input, test as seed_test, prediction

from .model_io import get_optimizer

logger = logging.getLogger("main")


def test(model, data, specific_class: int = None):
    """
    Evaluate accuracy using the model's full forward path.

    For *_MLP models (e.g., GCN_MLP), temporarily ensure the combined
    path (GCN + MLP) is active during evaluation by setting
    `mlp_freezed = False`, then restore the original setting.
    """
    orig_flag = None
    try:
        if hasattr(model, 'mlp_freezed'):
            orig_flag = model.mlp_freezed
            model.mlp_freezed = False
        return seed_test(model, data, specific_class)
    finally:
        if orig_flag is not None:
            model.mlp_freezed = orig_flag


def finetune_mlp(config, model, whole_data, train_data, batch_size, iters):
    input = grab_input(train_data)
    model.eval()
    model.mlp_freezed = True
    with torch.no_grad():
        gnn_output = model(**input)
        model.gnn_output = model(**grab_input(whole_data)).cpu()
        log_gnn_output = F.log_softmax(gnn_output, dim=-1)
    model.freeze_module(train=False)
    opt = get_optimizer(config['pipeline_params'], model)
    logger.info('start finetuning MLP')
    s = time.time()
    torch.cuda.synchronize()
    for _ in tqdm(range(iters)):
        opt.zero_grad()
        idx = np.random.choice(train_data.num_nodes, batch_size)
        idx = torch.from_numpy(idx).to(gnn_output.device)
        MLP_output = model.MLP(train_data.x[idx])
        cur_batch_gnn_output = gnn_output[idx]
        log_prob = F.log_softmax(MLP_output + cur_batch_gnn_output, dim=-1)
        main_loss = F.cross_entropy(MLP_output + gnn_output[idx], train_data.y[idx])
        kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
        (kl_loss + main_loss).backward()
        opt.step()
    torch.cuda.synchronize()
    e = time.time()
    logger.info(f'fine tune MLP used: {e - s} sec.')


def finetune_gnn_mlp(config, model, whole_data, train_data):
    model.freeze_module(train=False)
    dataset = config['eval_params']['dataset']
    if dataset == 'flickr' or (dataset == 'reddit2' and config['pipeline_params']['model_name']) or \
        (dataset in ['amazoncomputers', 'amazonphoto', 'coauthorcs', 'coauthorphysics']):
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=512, iters=100)
    else:
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=32, iters=100)
    bef_edit_ft_results = test(model, whole_data)
    ft_train_acc, ft_valid_acc, ft_test_acc = bef_edit_ft_results
    logger.info(f'before edit, after fine tune, train acc {ft_train_acc}, valid acc {ft_valid_acc}, test acc {ft_test_acc}')
    return bef_edit_ft_results


def success_rate(model, idx, label, whole_data):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    model.eval()
    input = grab_input(whole_data)
    is_regression = bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        if is_regression:
            y_pred = out.squeeze(-1) if out.dim() == 2 and out.size(-1) == 1 else out
        else:
            y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        if is_regression:
            pred_all = out.squeeze(-1) if out.dim() == 2 and out.size(-1) == 1 else out
            y_pred = pred_all[idx]
        else:
            y_pred = out.argmax(dim=-1)[idx]
    if is_regression:
        label_f = label.to(y_pred.dtype)
        scale = torch.std(label_f) if label_f.numel() > 1 else torch.abs(label_f).mean()
        tol = torch.clamp(scale * 0.1, min=1e-3)
        success = float((torch.abs(y_pred - label_f) <= tol).float().mean().item())
    else:
        success = int(y_pred.eq(label).sum()) / label.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return success



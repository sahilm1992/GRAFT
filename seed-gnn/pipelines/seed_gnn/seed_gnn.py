import pdb
from tqdm import tqdm
from copy import deepcopy

import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph

from main_utils import set_seeds_all
from models.base import BaseModel
from edit_gnn.utils import test, success_rate, prediction
from pipelines.seed_gnn.utils import get_optimizer, edit, visualize_validation


def select_mixup_training_nodes(model: BaseModel,
                                whole_data: Data,
                                alpha: int,
                                num_samples:int = 0,
                                center_node_idx=None):
    bef_edit_logits = prediction(model, whole_data)
    is_regression = bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )
    if is_regression:
        if bef_edit_logits.dim() == 2 and bef_edit_logits.size(-1) == 1:
            bef_edit_logits = bef_edit_logits.squeeze(-1)
        train_y_pred = bef_edit_logits[whole_data.train_mask]
        train_y_true = whole_data.y[whole_data.train_mask].to(train_y_pred.dtype)
        residual = torch.abs(train_y_pred - train_y_true)
        thresh = torch.quantile(residual, 0.5)
        right_pred_set = (residual <= thresh).nonzero(as_tuple=False)
    else:
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        train_y_true = whole_data.y[whole_data.train_mask]
        train_y_pred = bef_edit_pred[whole_data.train_mask]
        right_pred_set = train_y_pred.eq(train_y_true).nonzero(as_tuple=False)
    nodes_set = whole_data.train_mask.nonzero().squeeze()
    mixup_training_samples_idx = None
    if right_pred_set.numel() == 0:
        right_pred_set = torch.arange(nodes_set.numel(), device=nodes_set.device).view(-1, 1)
    dvc = right_pred_set.device

    neighbors = torch.Tensor([])
    num_hop = 0
    while len(neighbors) < num_samples and num_hop < 4:
        num_hop += 1
        neighbors, _, _, _ = k_hop_subgraph(center_node_idx, num_hops=num_hop, edge_index=whole_data.edge_index)
    
    # Choose neighbors for mixup
    correct_neighbors_pred_set = train_y_pred.eq(train_y_true).nonzero().to(dvc)
    correct_neighbors_pred_set = correct_neighbors_pred_set.squeeze().cpu().numpy().tolist()
    correct_neighbors_pred_set = torch.Tensor([int(i) for i in correct_neighbors_pred_set if i in neighbors]).unsqueeze(dim=1).type(torch.LongTensor).to(dvc)
    train_mixup_training_samples_idx = torch.cat((
                                                correct_neighbors_pred_set[torch.randperm(len(correct_neighbors_pred_set))[:int(num_samples * alpha)]].type(torch.LongTensor).to(dvc),
                                                right_pred_set[torch.randperm(len(right_pred_set))[int(num_samples * alpha):num_samples]]), dim=0)

    mixup_training_samples_idx = nodes_set[train_mixup_training_samples_idx]
    mixup_label = whole_data.y[mixup_training_samples_idx]

    return mixup_training_samples_idx, mixup_label


def seed_gnn_edit(config, model, node_idx_2flip, flipped_label, whole_data, max_num_step, debug=False):
    model.train()
    original_model = model
    model = deepcopy(model)
    optimizer = get_optimizer(config['pipeline_params'], model)
    raw_results = []
    mixup_training_samples_idx = None

    for idx in tqdm(range(len(node_idx_2flip))):
        set_seeds_all(idx)
        mixup_training_samples_idx, mixup_label = select_mixup_training_nodes(original_model, 
                                                                            whole_data,
                                                                            alpha = config['pipeline_params']['alpha'],
                                                                            num_samples = config['pipeline_params']['beta'],
                                                                            center_node_idx=node_idx_2flip[idx])
        
        nodes = torch.Tensor([])
        labels = torch.Tensor([])
        nodes = torch.cat((node_idx_2flip[:idx+1].squeeze(dim=1), mixup_training_samples_idx.squeeze(dim=1)), dim=0)
        labels = torch.cat((flipped_label[:idx+1].squeeze(dim=1), mixup_label.squeeze(dim=1)), dim=0)

        edited_model, success, loss, steps, mem, tot_time = edit(model,
                                                                whole_data,
                                                                nodes,
                                                                labels,
                                                                optimizer,
                                                                max_num_step,
                                                                num_edit_targets=idx + 1,
                                                                curr_edit_target=idx,
                                                                debug=debug)

        success = success_rate(model, node_idx_2flip[:idx+1].squeeze(dim=1), flipped_label[:idx+1].squeeze(dim=1), whole_data)
        res = [*test(edited_model, whole_data), success, steps]
        res.append(mem)
        res.append(tot_time)
        raw_results.append(res)

    visualize_validation(config,edited_model, whole_data, node_idx_2flip, " (SEED-GNN)")
    
    return raw_results

import time
import logging

import torch
import torch.nn.functional as F



from edit_gnn.utils import grab_input

logger = logging.getLogger("main")


def _as_1d_tensor(x: torch.Tensor) -> torch.Tensor:
    # Per-target loops may pass 0-D tensors; normalize to shape [1].
    if torch.is_tensor(x) and x.dim() == 0:
        return x.view(1)
    return x


def bef_edit_check(model, whole_data, idx, label, curr_edit_target):
    idx = _as_1d_tensor(idx)
    label = _as_1d_tensor(label)
    model.eval()
    torch.cuda.synchronize()
    input = grab_input(whole_data)
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        y_pred = out.argmax(dim=-1)[idx]
    if label.shape[0] == 1:
        success = (y_pred == label)
    else:
        success = 1.0 if y_pred.eq(label)[curr_edit_target] else 0.0
    torch.cuda.synchronize()
    return success


def single_edit(model, whole_data, idx, label, optimizer, max_num_step, num_edit_targets=1):
    idx = _as_1d_tensor(idx)
    label = _as_1d_tensor(label)
    s = time.time()
    loss_op = F.cross_entropy
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    success = False
    for step in range(1, max_num_step + 1):
        optimizer.zero_grad()
        input = grab_input(whole_data)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
            out = model.fast_forward(input['x'][idx], idx)
            loss = loss_op(out, label)
            y_pred = out.argmax(dim=-1)
        else:
            out = model(**input)
            loss = loss_op(out[idx], label)
            y_pred = out.argmax(dim=-1)[idx]
        loss.backward()
        optimizer.step()
        if label.shape[0] == 1:
            success = y_pred == label
        else:
            success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets])[:num_edit_targets].sum()) / num_edit_targets
        if success == 1.:
            break
    torch.cuda.synchronize()
    e = time.time()
    logger.info(f'max allocated mem: {torch.cuda.max_memory_allocated() / (1024**2)} MB')
    logger.info(f'edit time: {e - s}')
    return model, success, loss, step, torch.cuda.max_memory_allocated() / (1024**2), e - s


def edit(model, whole_data, idx, f_label, optimizer, max_num_step, num_edit_targets=1, curr_edit_target=0):
    bef = bef_edit_check(model, whole_data, idx, f_label, curr_edit_target=curr_edit_target)
    if bef == 1.:
        return model, bef, 0, 0, 0, 0
    return single_edit(model, whole_data, idx, f_label, optimizer, max_num_step, num_edit_targets=num_edit_targets)



import logging
from copy import deepcopy
from typing import Dict, List, Any, Tuple
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from editing_pipelines.editors.base import BaseEditor
from editing_pipelines.utils.model_io import get_optimizer
from editing_pipelines.utils.train_eval import test, success_rate
from editing_pipelines.utils.selection import select_edit_targets_by_strategy
from editing_pipelines.utils.visualization import plot_misclassification_by_attributes_before_after, plot_targeted_edits_distribution, plot_validation_correct_confidence_histogram
from editing_pipelines.utils.results import save_misclassifications_txt, save_misclassification_summary_txt

logger = logging.getLogger("main")
import time

class EWCEditor(BaseEditor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # self.ewc_lambda = config.get("ewc_params", {}).get("lambda", 1000)
        self.ewc_lambda = 1e6
        self.train_proxy_nodes = None
        self.train_proxy_labels = None
        logger.info(f"Initialized EWC Editor with λ = {self.ewc_lambda}")

    def select_edit_targets(self, **kwargs):
        num_targets = kwargs.get('num_targets', self.config['eval_params']['num_targets'])
        strategy = kwargs.get('strategy') or self.config.get('target_selection', {}).get('strategy', 'hard_misclassified_valid')
        node_idx_2flip, flipped_label = select_edit_targets_by_strategy(
            self.model, self.whole_data, self.num_classes, num_targets, strategy
        )
        return node_idx_2flip.cuda(), flipped_label.cuda()

    def select_train_proxy_nodes(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train_proxy_nodes, self.train_proxy_labels = select_edit_targets_by_strategy(
            self.model, self.whole_data, self.num_classes, num_samples, strategy='high_confidence_correct_valid'
        )
        return self.train_proxy_nodes.cuda(), self.train_proxy_labels.cuda()

    def compute_fisher_information(self, model, data: Data, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the diagonal Fisher Information (Ω) over provided node indices."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        from edit_gnn.utils import grab_input
        criterion = nn.CrossEntropyLoss()
        out = model(**grab_input(data))
        # Ensure 1-D indices on correct device
        if indices.dim() > 1:
            indices = indices.squeeze(dim=1)
        indices = indices.to(out.device)
        # Labels on device
        y_sel = data.y[indices].to(out.device)
        loss = criterion(out[indices], y_sel)
        model.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        return fisher

    def ewc_loss(self, model, fisher_dict, old_params, base_loss):
        """Compute total loss = new task loss + EWC penalty."""
        ewc_penalty = 0.0
        for n, p in model.named_parameters():
            if n in fisher_dict:
                ewc_penalty += torch.sum(fisher_dict[n] * (p - old_params[n]) ** 2)
        loss = base_loss + (self.ewc_lambda / 2) * ewc_penalty
        print(f"EWC penalty: {ewc_penalty}")
        print(f"Base loss: {base_loss}")
        print(f"Total loss: {loss}")
        return loss

    def edit_model(self, **kwargs) -> List[List[Any]]:
        node_idx_2flip: torch.Tensor = kwargs['node_idx_2flip']
        flipped_label: torch.Tensor = kwargs['flipped_label']
        max_num_step: int = kwargs.get('max_num_step', self.config['pipeline_params']['max_num_edit_steps'])
        logger.info("Starting EWC fine-tuning process")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Keep immutable before-edit copy; train on a working copy
        before_model = deepcopy(self.model).to(device).eval()
        model = deepcopy(self.model).to(device)
        print(f"Max number of steps: {max_num_step}")

        # Step 1: compute Fisher Ω on training proxy nodes (j=0)
        logger.info("Computing Fisher Information from proxy nodes (high-confidence correct)")
        # Select proxy nodes if not already selected
        num_proxy = kwargs.get('num_proxy', max(500, int(node_idx_2flip.numel())))
        proxy_nodes, _ = self.select_train_proxy_nodes(num_proxy)
        fisher_dict = self.compute_fisher_information(model, self.whole_data, proxy_nodes)
        old_params = {n: p.clone().detach() for n, p in model.named_parameters()}

        # Step 2: fine-tune model with EWC loss on edit targets (j=1)
        logger.info("Fine-tuning on edit targets with EWC regularization")
        optimizer = get_optimizer(self.config["pipeline_params"], model)
        criterion = nn.CrossEntropyLoss()
        raw_results = []
        model.train()
        start_time = time.time()
        # Ensure 1D shapes for indexing/labels
        if node_idx_2flip.dim() > 1:
            node_idx_2flip = node_idx_2flip.squeeze(dim=1)
        if flipped_label.dim() > 1:
            flipped_label = flipped_label.squeeze(dim=1)

        from edit_gnn.utils import grab_input
        for epoch in tqdm(range(max_num_step), desc="EWC fine-tuning"):
            optimizer.zero_grad()
            out = model(**grab_input(self.whole_data))
            base_loss = criterion(out[node_idx_2flip], flipped_label)
            total_loss = self.ewc_loss(model, fisher_dict, old_params, base_loss)
            total_loss.backward()
            optimizer.step()
            # epoch eval
            results_after = test(model, self.whole_data)
            current_success = success_rate(model, node_idx_2flip, flipped_label, self.whole_data)
            mem_usage = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # MB approx
            steps = max_num_step
            total_time = time.time() - start_time
            res = [*results_after, current_success, steps, mem_usage, total_time]
            raw_results.append(res)
        logger.info(f"EWC editing completed: {res}")

        save_misclassifications_txt(
            self.config,
            model_before=before_model,
            model_after=model,
            whole_data=self.whole_data,
            method_name='ewc',
            model_name=self.config['pipeline_params']['model_name'],
            file_suffix=''
        )
        save_misclassification_summary_txt(
            self.config,
            model_before=before_model,
            model_after=model,
            whole_data=self.whole_data,
            method_name='ewc',
            model_name=self.config['pipeline_params']['model_name'],
             file_suffix='',
             edit_indices=node_idx_2flip
        )
        # plot_misclassification_by_attributes_before_after(
        #     self.config,
        #     model_before=before_model,
        #     model_after=model,
        #     whole_data=self.whole_data,
        #     method_name='ewc',
        #     model_name=self.config['pipeline_params']['model_name'],
        #     file_suffix=''
        # )
        # plot_validation_correct_confidence_histogram(
        #      self.config,
        #      model_before=before_model,
        #      model_after=model,
        #      whole_data=self.whole_data,
        #      method_name='ewc',
        #      model_name=self.config['pipeline_params']['model_name'],
        #      file_suffix=''
        #  )
        # plot_targeted_edits_distribution(
        #      self.config,
        #      edited_node_idx=node_idx_2flip,
        #      whole_data=self.whole_data,
        #      method_name='ewc',
        #      model_name=self.config['pipeline_params']['model_name'],
        #      file_suffix=''
        #  )
        self.model = model
        return raw_results

import copy
import inspect

import torch
from torch import Tensor
from torch_sparse import SparseTensor
try:
    from torch_geometric.nn.models import Polynormer as PyGPolynormer
except ImportError:
    PyGPolynormer = None

from .base import BaseGNNModel
from .mlp import MLP

POLYNORMER_IMPORT_ERROR = (
    "torch_geometric.nn.models.Polynormer is not available in the current "
    "PyG installation. Please upgrade PyG to a version that provides Polynormer."
)


class Polynormer(BaseGNNModel):
    """Wrapper around PyG's Polynormer that returns raw logits.

    PyGPolynormer.forward ends with ``F.log_softmax(x, dim=-1)`` which is
    incompatible with regression (out_channels=1 collapses to constant 0) and
    suboptimal for classification when paired with ``F.cross_entropy`` (which
    already includes softmax).  We register forward hooks on the prediction
    heads (``pred_local`` / ``pred_global``) to capture the raw logits before
    ``log_softmax`` is applied and return those instead.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual: bool = False,
        load_pretrained_backbone: bool = False,
        saved_ckpt_path: str = "",
        **kwargs,
    ):
        if PyGPolynormer is None:
            raise ImportError(POLYNORMER_IMPORT_ERROR)
        super(Polynormer, self).__init__(
            in_channels, hidden_channels, out_channels, num_layers, dropout, batch_norm, residual
        )
        model_kwargs = self._filter_polynormer_kwargs(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
        self.model = PyGPolynormer(**model_kwargs)
        if hasattr(self.model, "convs"):
            self.convs = self.model.convs

        self._raw_logits: Tensor | None = None
        self._install_logit_hooks()

    def _install_logit_hooks(self):
        """Hook pred_local and pred_global to capture raw logits
        (before PyGPolynormer applies log_softmax)."""
        # Remove any stale hooks first (important after deepcopy).
        for name in ("pred_local", "pred_global"):
            head = getattr(self.model, name, None)
            if head is not None:
                head._forward_hooks.clear()

        def _capture(_, _input, output):
            self._raw_logits = output

        for name in ("pred_local", "pred_global"):
            head = getattr(self.model, name, None)
            if head is not None:
                head.register_forward_hook(_capture)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        result._raw_logits = None
        result._install_logit_hooks()
        return result

    @staticmethod
    def _filter_polynormer_kwargs(**candidate_kwargs):
        if PyGPolynormer is None:
            raise ImportError(POLYNORMER_IMPORT_ERROR)
        sig = inspect.signature(PyGPolynormer.__init__)
        valid = set(sig.parameters.keys())
        valid.discard("self")
        return {k: v for k, v in candidate_kwargs.items() if k in valid}

    @staticmethod
    def _to_edge_index(adj_t: SparseTensor):
        if isinstance(adj_t, torch.Tensor):
            return adj_t
        if hasattr(adj_t, "t") and callable(adj_t.t):
            row, col, _ = adj_t.t().coo()
            return torch.stack([row, col], dim=0)
        row, col, _ = adj_t.coo()
        return torch.stack([row, col], dim=0)

    @staticmethod
    def _resolve_batch(x: Tensor, kwargs):
        batch = kwargs.get("batch", None)
        if batch is None:
            # Single whole-graph setting: every node belongs to graph 0.
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return batch

    def reset_parameters(self):
        if hasattr(self.model, "reset_parameters"):
            self.model.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        self._raw_logits = None
        try:
            self.model(x, adj_t, *args, **kwargs)
        except TypeError:
            edge_index = self._to_edge_index(adj_t)
            batch = self._resolve_batch(x, kwargs)
            self.model(x, edge_index, batch, *args, **kwargs)
        return self._raw_logits


class Polynormer_MLP(BaseGNNModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        shared_weights: bool = True,
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual: bool = False,
        load_pretrained_backbone: bool = False,
        saved_ckpt_path: str = "",
        **kwargs,
    ):
        super(Polynormer_MLP, self).__init__(
            in_channels, hidden_channels, out_channels, num_layers, dropout, batch_norm, residual
        )

        if load_pretrained_backbone:
            self.Polynormer = Polynormer.from_pretrained(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                saved_ckpt_path=saved_ckpt_path,
                num_layers=num_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                **kwargs,
            )
        else:
            self.Polynormer = Polynormer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                **kwargs,
            )

        self.MLP = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
        )

        self.mlp_freezed = True
        if load_pretrained_backbone:
            self.freeze_layer(self.Polynormer, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)
        self.gnn_output = None

    def reset_parameters(self):
        self.Polynormer.reset_parameters()
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()

    def freeze_layer(self, model, freeze=True):
        for _, p in model.named_parameters():
            p.requires_grad = not freeze

    def freeze_module(self, train=True):
        if train:
            self.freeze_layer(self.Polynormer, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.Polynormer, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        polynormer_out = self.Polynormer(x, adj_t, *args, **kwargs)
        if self.mlp_freezed:
            return polynormer_out
        mlp_out = self.MLP(x, *args)
        return polynormer_out + mlp_out

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + self.MLP(x)

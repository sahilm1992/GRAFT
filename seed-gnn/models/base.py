import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from pathlib import Path
import pdb
import inspect

from tqdm import tqdm


class BaseModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False):

        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_layers = num_layers
        self.use_linear = use_linear
        if self.batch_norm:
            self.bns = ModuleList()
            for _ in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)


    @classmethod
    def from_pretrained(cls, in_channels: int, out_channels: int, saved_ckpt_path: str, **kwargs):
        # Identify valid arguments for the constructor to avoid TypeErrors when extra metadata is passed
        sig = inspect.signature(cls.__init__)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kwargs:
            # Preserve all kwargs for constructors that intentionally route
            # optional behavior through **kwargs (e.g., gat_optimized).
            init_kwargs = dict(kwargs)
        else:
            init_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Ensure saved_ckpt_path is passed to constructor if it expects it
        if 'saved_ckpt_path' in sig.parameters:
            init_kwargs['saved_ckpt_path'] = saved_ckpt_path

        model = cls(in_channels=in_channels, out_channels=out_channels, **init_kwargs)
        
        if not saved_ckpt_path.endswith('.pt'):
            # 1. Base globbing based on class name
            checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_*.pt")]
            
            # Handle Lora models which might use the base class checkpoint
            if '_Lora' in cls.__name__:
                checkpoints.extend(str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__.replace('_Lora', '')}_*.pt"))
            
            glob_checkpoints = checkpoints
            
            # 2. Filter out MLP checkpoints if the current class is not an MLP
            if '_MLP' not in cls.__name__:
                glob_checkpoints = [x for x in glob_checkpoints if '_MLP' not in x]
            
            # 3. Narrow down by num_layers if it's in kwargs and present in filenames
            num_layers = kwargs.get('num_layers')
            if num_layers is not None:
                layer_pattern = f"_layers{num_layers}_"
                filtered = [x for x in glob_checkpoints if layer_pattern in x]
                if filtered:
                    glob_checkpoints = filtered
            
            # 4. Narrow down by seed if it's in kwargs and present in filenames
            seed = kwargs.get('seed')
            if seed is not None:
                # Support both _seed{seed}_ and _seed{seed}.pt formats
                seed_pattern = f"seed{seed}"
                filtered = [x for x in glob_checkpoints if f"_{seed_pattern}_" in x or f"_{seed_pattern}.pt" in x or f"_{seed_pattern}_" in x]
                if filtered:
                    glob_checkpoints = filtered
            
            # 5. If multiple found, prefer the most specific match (both layers and seed)
            if len(glob_checkpoints) > 1:
                if num_layers is not None and seed is not None:
                    both = [x for x in glob_checkpoints if f"_layers{num_layers}_" in x and (f"_seed{seed}_" in x or f"_seed{seed}.pt" in x)]
                    if len(both) == 1:
                        glob_checkpoints = both

            assert len(glob_checkpoints) == 1, (
                f"Expected 1 checkpoint, found {len(glob_checkpoints)} for {cls.__name__} in {saved_ckpt_path}. "
                f"Metadata: layers={num_layers}, seed={seed}. Found: {glob_checkpoints}"
            )
            saved_ckpt_path = glob_checkpoints[0]
            
        print(f'load model weights from {saved_ckpt_path}')
        state_dict = torch.load(saved_ckpt_path, map_location='cpu')
        final_state_dict = {}
        target_keys = set(model.state_dict().keys())
        remap_hits = 0
        ignore_keys = ['edit_lrs']
        for raw_k, v in state_dict.items():
            if raw_k in ignore_keys:
                continue

            # Handle namespace differences between wrapped models (expects
            # "model.*") and legacy checkpoints (often saved as plain "*").
            candidates = [raw_k]
            if raw_k.startswith('model.'):
                candidates.append(raw_k[len('model.'):])
            else:
                candidates.append(f"model.{raw_k}")

            mapped_key = None
            for candidate in candidates:
                if candidate in target_keys:
                    mapped_key = candidate
                    break

            if mapped_key is None:
                # Keep original behavior as fallback; strict=False will report it.
                mapped_key = raw_k
            elif mapped_key != raw_k:
                remap_hits += 1

            final_state_dict[mapped_key] = v
        incompatible = model.load_state_dict(final_state_dict, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
        if remap_hits:
            print(f"[from_pretrained] remapped {remap_hits} checkpoint keys to match model namespace")
        if missing_keys or unexpected_keys:
            print(
                f"[from_pretrained] checkpoint compatibility for {cls.__name__}: "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )
            if missing_keys:
                print(f"[from_pretrained] missing keys (first 20): {missing_keys[:20]}")
            if unexpected_keys:
                print(f"[from_pretrained] unexpected keys (first 20): {unexpected_keys[:20]}")
        else:
            print(f"[from_pretrained] checkpoint compatibility for {cls.__name__}: perfect key match")
        return model


    def reset_parameters(self):
        raise NotImplementedError


class BaseGNNModel(BaseModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False):
        super(BaseGNNModel, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                                           dropout, batch_norm, residual, use_linear)
        if self.use_linear:
            self.lins = torch.nn.ModuleList()
        self.convs = ModuleList()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        for idx in range(self.num_layers - 1):
            conv = self.convs[idx]
            h = conv(x, adj_t)
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)
        h = self.convs[-1](x, adj_t, *args, **kwargs)
        if self.use_linear:
            linear = self.lins[-1](x)
            x = h + linear
        else:
            x = h
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, size):
        if self.use_linear:
            raise NotImplementedError
        if layer != 0:
            x = self.dropout(x)
        x_target = x[:size[1]]
        h = self.convs[layer]((x, x_target), adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = F.relu(h)
        return h


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        for i in range(len(self.convs)):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index, size).cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

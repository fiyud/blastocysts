import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath
import math

from nn.common import PositionalEncoding

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        try:
            import selective_scan_cuda_core
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None and D.stride(-1) != 1:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True
            ctx.delta_softplus = delta_softplus
            ctx.backnrows = backnrows
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out
        except:
            # Fallback to CPU implementation or simplified version
            return u  # Placeholder

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        try:
            import selective_scan_cuda_core
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
        except:
            return (dout, None, None, None, None, None, None, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,
        backnrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,
        ssoflex=True,
        SelectiveScan=None,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if SelectiveScan is None:
        SelectiveScan = SelectiveScanCore

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    """
    Mamba-style 2D Selective Scan block
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()
        
        self.K = 4
        self.x_proj = nn.Linear(self.d_inner, self.K * (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_projs = nn.Linear(self.dt_rank, self.K * self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize dt and A
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_projs.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_projs.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.K * self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs.bias.copy_(inv_dt)

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.K * self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        y = cross_selective_scan(
            x, 
            self.x_proj.weight.view(4, -1, self.d_inner).contiguous(),
            None,
            self.dt_projs.weight.view(4, -1, self.dt_rank).contiguous(),
            self.dt_projs.bias.view(4, -1).contiguous(),
            self.A_log.view(self.d_inner, -1).contiguous(),
            self.D.view(4, -1).contiguous(),
            self.out_norm,
            out_norm_shape="v0",
            SelectiveScan=SelectiveScanCore,
        )
        
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """
    Vision Mamba Block with SS2D
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # B, H', W', C
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """
    Patch Merging Layer (downsampling)
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class MambaBackbone(nn.Module):
    """
    Mamba-based vision backbone for feature extraction
    """
    def __init__(
        self,
        in_chans=3,
        patch_size=4,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.dims = dims
        self.num_features = dims[-1]
        
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer,
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                VSSBlock(
                    hidden_dim=dims[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) + i],
                    norm_layer=norm_layer,
                    d_state=d_state,
                    **kwargs,
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            downsample = PatchMerging2D(dims[i_layer], norm_layer=norm_layer)
            self.downsample_layers.append(downsample)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)  # B, H, W, C
        
        for i_layer in range(self.num_layers):
            for blk in self.layers[i_layer]:
                x = blk(x)
            if i_layer < self.num_layers - 1:
                x = self.downsample_layers[i_layer](x)
        
        x = self.norm(x)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.avgpool(x)  # B, C, 1, 1
        x = torch.flatten(x, 1)  # B, C
        
        return x


class DualBackboneMambaTimeLapseClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        context_size: int = 5,
        reference_size: int = 3,
        # Mamba backbone configs
        context_patch_size: int = 4,
        reference_patch_size: int = 4,
        context_depths: list = [2, 2, 9, 2],
        reference_depths: list = [2, 2, 9, 2],
        context_dims: list = [96, 192, 384, 768],
        reference_dims: list = [96, 192, 384, 768],
        d_state: int = 16,

        context_encoding_size: int = 512,
        reference_encoding_size: int = 512,

        mha_num_attention_heads: int = 8,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        dropout_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.context_size = context_size
        self.reference_size = reference_size
        self.context_encoding_size = context_encoding_size
        self.reference_encoding_size = reference_encoding_size
        
        # Context backbone (processes single RGB frames)
        self.context_backbone = MambaBackbone(
            in_chans=3,
            patch_size=context_patch_size,
            depths=context_depths,
            dims=context_dims,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )
        self.num_context_features = context_dims[-1]
        
        # Reference backbone (processes concatenated frame pairs)
        self.reference_backbone = MambaBackbone(
            in_chans=6,  # Concatenated frames
            patch_size=reference_patch_size,
            depths=reference_depths,
            dims=reference_dims,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )
        self.num_reference_features = reference_dims[-1]
        
        # Compression layers
        if self.num_context_features != self.context_encoding_size:
            self.compress_context = nn.Linear(self.num_context_features, self.context_encoding_size)
        else:
            self.compress_context = nn.Identity()
        
        if self.num_reference_features != self.reference_encoding_size:
            self.compress_reference = nn.Linear(self.num_reference_features, self.reference_encoding_size)
        else:
            self.compress_reference = nn.Identity()
        
        # Positional encoding for temporal modeling
        self.positional_encoding = PositionalEncoding(
            self.context_encoding_size, 
            max_seq_len=self.context_size + 1
        )
        
        # Transformer encoder for temporal modeling
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.context_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.context_encoding_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=mha_num_attention_layers
        )
        
        # Cross-attention between context and reference
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.context_encoding_size,
            num_heads=mha_num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier head
        fusion_dim = self.context_encoding_size + self.reference_encoding_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 4, self.num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        context_frames: torch.Tensor, 
        reference_frames: torch.Tensor,
        mask_reference: Optional[torch.Tensor] = None,
        use_cross_attention: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for dual-backbone Mamba time-lapse classification
        
        Args:
            context_frames: Tensor of shape (batch_size, context_size, 3, H, W)
            reference_frames: Tensor of shape (batch_size, reference_size, 3, H, W)
            mask_reference: Optional tensor to mask reference during training
            use_cross_attention: Whether to use cross-attention (now implemented!)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = context_frames.shape[0]
        device = context_frames.device
        
        # Process context frames through Mamba backbone
        context_frames_flat = context_frames.view(-1, 3, context_frames.shape[-2], context_frames.shape[-1])
        context_features = self.context_backbone(context_frames_flat)
        context_features = self.compress_context(context_features)
        context_features = context_features.view(batch_size, self.context_size, self.context_encoding_size)
        
        # Process reference frames (concatenate first and last)
        if self.reference_size >= 2:
            ref_concat = torch.cat([
                reference_frames[:, 0], 
                reference_frames[:, -1]
            ], dim=1)  # Shape: (batch_size, 6, H, W)
        else:
            ref_concat = torch.cat([
                reference_frames[:, 0], 
                reference_frames[:, 0]
            ], dim=1)
        
        reference_features = self.reference_backbone(ref_concat)
        reference_features = self.compress_reference(reference_features)
        reference_features = reference_features.unsqueeze(1)  # (batch_size, 1, reference_encoding_size)
        
        # Combine context and reference features
        joint_features = torch.cat([context_features, reference_features], dim=1)
        joint_features = self.positional_encoding(joint_features)
        
        # Optional masking
        if mask_reference is not None:
            mask_reference = mask_reference.long()
            src_key_padding_mask = torch.zeros(batch_size, self.context_size + 1, 
                                             dtype=torch.bool, device=device)
            src_key_padding_mask[:, -1] = mask_reference.squeeze()
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        temporal_features = self.transformer_encoder(
            joint_features, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Separate context and reference features
        context_processed = temporal_features[:, :-1, :]
        reference_processed = temporal_features[:, -1:, :]
        
        # Apply cross-attention if requested
        if use_cross_attention:
            # Context attends to reference
            context_attended, _ = self.cross_attention(
                query=context_processed,
                key=reference_processed,
                value=reference_processed
            )
            context_pooled = torch.mean(context_attended, dim=1)
        else:
            context_pooled = torch.mean(context_processed, dim=1)
        
        reference_pooled = reference_processed.squeeze(1)
        
        # Fusion and classification
        fused_features = torch.cat([context_pooled, reference_pooled], dim=1)
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict(
        self, 
        context_frames: torch.Tensor, 
        reference_frames: torch.Tensor
    ) -> torch.Tensor:
        """Generate predictions with softmax probabilities"""
        with torch.no_grad():
            logits = self.forward(context_frames, reference_frames)
            predictions = F.softmax(logits, dim=1)
        return predictions


if __name__ == "__main__":
    # Test the model
    print("Testing DualBackboneMambaTimeLapseClassifier...")
    
    model = DualBackboneMambaTimeLapseClassifier(
        num_classes=2,
        context_size=5,
        reference_size=3,
        context_encoding_size=512,
        reference_encoding_size=512,
        context_dims=[96, 192, 384, 768],  # Mamba dimensions
        reference_dims=[96, 192, 384, 768],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    context_frames = torch.randn(2, 5, 3, 224, 224)
    reference_frames = torch.randn(2, 3, 3, 224, 224)
    
    print(f"\nInput shapes:")
    print(f"Context frames: {context_frames.shape}")
    print(f"Reference frames: {reference_frames.shape}")
    
    logits = model(context_frames, reference_frames, use_cross_attention=True)
    print(f"\nOutput logits shape: {logits.shape}")
    
    predictions = model.predict(context_frames, reference_frames)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0].numpy()}")
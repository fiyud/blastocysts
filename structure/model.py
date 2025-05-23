import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from efficientnet_pytorch import EfficientNet
from structure.nn.common import PositionalEncoding

class DualBackboneTimeLapseClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        context_size: int = 5,  # temporal context frames
        reference_size: int = 3,  # reference frames
        context_encoder: str = "efficientnet-b0",
        reference_encoder: str = "efficientnet-b0", 
        context_encoding_size: int = 512,
        reference_encoding_size: int = 512,
        mha_num_attention_heads: int = 8,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.context_size = context_size
        self.reference_size = reference_size
        self.context_encoding_size = context_encoding_size
        self.reference_encoding_size = reference_encoding_size
        
        if context_encoder.split("-")[0] == "efficientnet":
            self.context_backbone = EfficientNet.from_name(context_encoder, in_channels=3)
            self.context_backbone = replace_bn_with_gn(self.context_backbone)
            self.num_context_features = self.context_backbone._fc.in_features
        else:
            raise NotImplementedError(f"Context encoder {context_encoder} not implemented")
        
        if reference_encoder.split("-")[0] == "efficientnet":
            self.reference_backbone = EfficientNet.from_name(reference_encoder, in_channels=6)  # concat 2 frames
            self.reference_backbone = replace_bn_with_gn(self.reference_backbone)
            self.num_reference_features = self.reference_backbone._fc.in_features
        else:
            raise NotImplementedError(f"Reference encoder {reference_encoder} not implemented")
        
        if self.num_context_features != self.context_encoding_size:
            self.compress_context = nn.Linear(self.num_context_features, self.context_encoding_size)
        else:
            self.compress_context = nn.Identity()
        
        if self.num_reference_features != self.reference_encoding_size:
            self.compress_reference = nn.Linear(self.num_reference_features, self.reference_encoding_size)
        else:
            self.compress_reference = nn.Identity()
        
        self.positional_encoding = PositionalEncoding(
            self.context_encoding_size, 
            max_seq_len=self.context_size + 1
        )
        
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
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.context_encoding_size,
            num_heads=mha_num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        fusion_dim = self.context_encoding_size + self.reference_encoding_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 4, self.num_classes)
        )
        
        # Attention mask for reference token (optional masking during training)
        self.register_buffer('reference_mask', torch.zeros((1, self.context_size + 1), dtype=torch.bool))
        self.register_buffer('no_mask', torch.zeros((1, self.context_size + 1), dtype=torch.bool))
        
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
        mask_reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for dual-backbone time-lapse classification
        
        Args:
            context_frames: Tensor of shape (batch_size, context_size, 3, H, W)
                           Sequential frames showing temporal progression
            reference_frames: Tensor of shape (batch_size, reference_size, 3, H, W)
                             Reference/baseline frames for comparison
            mask_reference: Optional tensor to mask reference during training
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = context_frames.shape[0]
        device = context_frames.device
        
        context_frames_flat = context_frames.view(-1, 3, context_frames.shape[-2], context_frames.shape[-1])
        context_features = self.context_backbone.extract_features(context_frames_flat)
        context_features = self.context_backbone._avg_pooling(context_features)
        
        if self.context_backbone._global_params.include_top:
            context_features = context_features.flatten(start_dim=1)
            context_features = self.context_backbone._dropout(context_features)
        
        context_features = self.compress_context(context_features)
        context_features = context_features.view(batch_size, self.context_size, self.context_encoding_size)
        
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
        
        reference_features = self.reference_backbone.extract_features(ref_concat)
        reference_features = self.reference_backbone._avg_pooling(reference_features)
        
        if self.reference_backbone._global_params.include_top:
            reference_features = reference_features.flatten(start_dim=1)
            reference_features = self.reference_backbone._dropout(reference_features)
        
        reference_features = self.compress_reference(reference_features)
        reference_features = reference_features.unsqueeze(1)  # (batch_size, 1, reference_encoding_size)
        
        joint_features = torch.cat([context_features, reference_features], dim=1)
        
        joint_features = self.positional_encoding(joint_features)
        
        if mask_reference is not None:
            mask_reference = mask_reference.long()
            # Create attention mask (True means mask out)
            src_key_padding_mask = torch.zeros(batch_size, self.context_size + 1, 
                                             dtype=torch.bool, device=device)
            src_key_padding_mask[:, -1] = mask_reference.squeeze()  # Mask reference token
        else:
            src_key_padding_mask = None
        
        temporal_features = self.transformer_encoder(
            joint_features, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Separate context and reference features after transformer
        context_processed = temporal_features[:, :-1, :]  # All but last token
        reference_processed = temporal_features[:, -1:, :]  # Last token
        
        context_pooled = torch.mean(context_processed, dim=1)
        reference_pooled = reference_processed.squeeze(1)
        
        fused_features = torch.cat([context_pooled, reference_pooled], dim=1)
        
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict(
        self, 
        context_frames: torch.Tensor, 
        reference_frames: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(context_frames, reference_frames)
            predictions = F.softmax(logits, dim=1)
        return predictions


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16
) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features
        )
    )
    return root_module


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """Replace all submodules selected by the predicate with the output of func."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        
        tgt_module = func(src_module)
        
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    
    return root_module


if __name__ == "__main__":
    model = DualBackboneTimeLapseClassifier(
        num_classes=2,
        context_size=5,
        reference_size=3,
        context_encoding_size=512,
        reference_encoding_size=512
    )
    
    # Example input
    context_frames = torch.randn(4, 5, 3, 224, 224)  # batch=4, context=5 frames
    reference_frames = torch.randn(4, 3, 3, 224, 224)  # batch=4, reference=3 frames
    
    logits = model(context_frames, reference_frames)
    print(f"Output shape: {logits.shape}")  # Should be (4, 2)
    
    predictions = model.predict(context_frames, reference_frames)
    print(f"Predictions shape: {predictions.shape}")  # Should be (4, 2)
    print(f"Sample predictions: {predictions[0].numpy()}")  # Probabilities for first sample
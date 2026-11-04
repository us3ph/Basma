"""BasmaDetector: a DINOv2 (with registers) encoder, a global head on CLS + mean patch token and a per-patch head
pooled with a top-quantile mean. One logit per view; the fusion layer combines views (fusion.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class BasmaDetector(nn.Module):
    def __init__(self, backbone="facebook/dinov2-with-registers-base", quantile=0.25, dropout=0.1, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone)
        else:
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        cfg = self.backbone.config
        h = cfg.hidden_size
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0))
        self.quantile = quantile
        self.head_g = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1))
        self.head_p = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        nn.init.zeros_(self.head_g[-1].bias); nn.init.zeros_(self.head_p[-1].bias)

    def freeze_blocks(self, n):
        """Freeze the embeddings and the first n encoder layers (0 = full fine-tuning)."""
        if n <= 0:
            return
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for layer in self.backbone.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def param_groups(self, lr_backbone, lr_head, layer_decay=0.85, weight_decay=0.05):
        layers = self.backbone.encoder.layer
        n = len(layers)
        groups = []
        def add(params, lr):
            params = [p for p in params if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
        add(self.backbone.embeddings.parameters(), lr_backbone * layer_decay ** n)
        for i, layer in enumerate(layers):
            add(layer.parameters(), lr_backbone * layer_decay ** (n - 1 - i))
        add(self.backbone.layernorm.parameters(), lr_backbone)
        add(list(self.head_g.parameters()) + list(self.head_p.parameters()), lr_head)
        return groups

    def forward(self, x):
        """x (N, 3, 224, 224) -> dict(logit (N,), logit_g (N,), logit_p (N,), patch (N, P))."""
        out = self.backbone(pixel_values=x).last_hidden_state
        cls = out[:, 0]
        patches = out[:, 1 + self.n_reg:]
        g = self.head_g(torch.cat([cls, patches.mean(1)], dim=-1)).squeeze(-1)
        pl = self.head_p(patches).squeeze(-1)
        k = max(1, int(self.quantile * int(pl.shape[1]) + 0.5))      # int() also works on traced shapes (ONNX export)
        p = pl.topk(k, dim=1).values.mean(1)
        return {"logit": g + p, "logit_g": g, "logit_p": p, "patch": pl}


def detector_loss(out, y, aux=0.25):
    """BCE on the fused logit plus auxiliary BCE on each head so both stay informative on their own."""
    main = F.binary_cross_entropy_with_logits(out["logit"], y)
    a = F.binary_cross_entropy_with_logits(out["logit_g"], y) + F.binary_cross_entropy_with_logits(out["logit_p"], y)
    return main + aux * a, main

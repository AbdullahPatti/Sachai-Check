"""XLS-R 300M with a learned weighted sum over all hidden layers, attentive statistics pooling and a small MLP head.

Why this design:
  * XLS-R was pretrained on 128 languages including Urdu, so its features already model Urdu phonetics.
  * Spoofing cues sit mostly in lower and middle transformer layers, so we let the model learn a softmax
    weight per layer (the idea behind the SLS classifier, Zhang et al., ACM MM 2024) instead of using only
    the last layer. The learned weights are also a research finding in themselves.
  * Attentive statistics pooling (Okabe et al., 2018) turns variable-length frame features into one clip vector.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

WEIGHTS_FILE = "model.safetensors"


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim: int, att_dim: int = 128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, att_dim), nn.Tanh(), nn.Linear(att_dim, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.softmax(self.att(x).squeeze(-1).float(), dim=1)          # (B, T)
        xf = x.float()
        mu = torch.sum(a.unsqueeze(-1) * xf, dim=1)
        var = torch.sum(a.unsqueeze(-1) * xf ** 2, dim=1) - mu ** 2
        sd = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mu, sd], dim=-1), a


class SachaiDetector(nn.Module):
    def __init__(self, ssl_config: Wav2Vec2Config, proj_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.ssl = Wav2Vec2Model(ssl_config)
        n_states = ssl_config.num_hidden_layers + 1
        hidden = ssl_config.hidden_size
        self.layer_logits = nn.Parameter(torch.zeros(n_states))
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.pool = AttentiveStatsPool(proj_dim)
        self.head = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_dim, 2))
        self.head_config = {"proj_dim": proj_dim, "dropout": dropout}

    # ------------------------------------------------------------ construction
    @classmethod
    def from_backbone(cls, name: str, proj_dim: int = 256, dropout: float = 0.2, mask_time_prob: float = 0.05,
                      grad_checkpointing: bool = True) -> "SachaiDetector":
        overrides = dict(layerdrop=0.0, mask_time_prob=mask_time_prob, mask_time_length=10, mask_feature_prob=0.0,
                         hidden_dropout=0.1, feat_proj_dropout=0.1, attention_dropout=0.0, activation_dropout=0.0)
        ssl = Wav2Vec2Model.from_pretrained(name, **overrides)
        model = cls(ssl.config, proj_dim, dropout)
        model.ssl = ssl
        model.prepare_for_training(grad_checkpointing)
        return model

    def prepare_for_training(self, grad_checkpointing: bool = True) -> None:
        self.ssl.freeze_feature_encoder()   # CNN waveform encoder stays frozen
        if grad_checkpointing:
            self.ssl.gradient_checkpointing_enable()

    def save_pretrained(self, out_dir: str | Path) -> None:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        tmp = out / (WEIGHTS_FILE + ".tmp")
        save_file(state, str(tmp))
        tmp.replace(out / WEIGHTS_FILE)
        self.ssl.config.to_json_file(str(out / "ssl_config.json"))
        (out / "head_config.json").write_text(json.dumps(self.head_config, indent=2))

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, map_location: str = "cpu") -> "SachaiDetector":
        from safetensors.torch import load_file

        d = Path(model_dir)
        ssl_cfg = Wav2Vec2Config.from_json_file(str(d / "ssl_config.json"))
        head = json.loads((d / "head_config.json").read_text())
        model = cls(ssl_cfg, **head)
        missing, unexpected = model.load_state_dict(load_file(str(d / WEIGHTS_FILE), device=map_location), strict=False)
        bad = [k for k in missing if "masked_spec_embed" not in k]
        if bad or unexpected:
            raise RuntimeError(f"Weight mismatch. missing={bad[:5]} unexpected={unexpected[:5]}")
        return model

    # ------------------------------------------------------------ forward
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.float(), dim=0)

    def forward(self, wav: torch.Tensor) -> dict:
        out = self.ssl(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)                             # (L, B, T, H)
        w = self.layer_weights().to(hs.dtype)
        x = torch.einsum("l,lbth->bth", w, hs)
        x = self.proj(self.norm(x))
        emb, att = self.pool(x)
        logits = self.head(emb.to(x.dtype)).float()
        return {"logits": logits, "embedding": emb, "attention": att}

    @staticmethod
    def spoof_logit(logits: torch.Tensor) -> torch.Tensor:
        """Log-odds of 'synthetic'. Positive means synthetic is more likely."""
        return logits[:, 1] - logits[:, 0]


def soft_ce(logits: torch.Tensor, y: torch.Tensor, smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits, y, label_smoothing=smoothing)

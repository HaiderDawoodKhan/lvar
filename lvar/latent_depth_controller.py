from typing import Optional

import torch
import torch.nn as nn


class LatentDepthController(nn.Module):
    """
    Binary STOP/CONTINUE controller over image summary, prompt tail, and latents.

    The scalar output is a STOP logit: positive values favor stopping, negative
    values favor continuing recurrent latent reasoning.
    """

    TYPE_CLS = 0
    TYPE_IMAGE = 1
    TYPE_PROMPT = 2
    TYPE_LATENT = 3

    def __init__(
        self,
        input_hidden_size: int,
        controller_hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        max_prompt_tokens: int = 10,
        max_latent_steps: int = 10,
    ) -> None:
        super().__init__()
        self.input_hidden_size = int(input_hidden_size)
        self.controller_hidden_size = int(controller_hidden_size)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_latent_steps = int(max_latent_steps)
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive.")
        if self.max_latent_steps < 0:
            raise ValueError("max_latent_steps must be non-negative.")

        self.input_proj = nn.Linear(self.input_hidden_size, self.controller_hidden_size)
        self.cls_token = nn.Parameter(torch.randn(self.controller_hidden_size) * 0.02)
        self.type_embedding = nn.Embedding(4, self.controller_hidden_size)
        self.position_embedding = nn.Embedding(
            2 + self.max_prompt_tokens + self.max_latent_steps,
            self.controller_hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.controller_hidden_size,
            nhead=int(num_heads),
            dim_feedforward=self.controller_hidden_size * int(ff_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.controller_hidden_size)
        self.stop_head = nn.Linear(self.controller_hidden_size, 1)

    def forward(
        self,
        visual_token: torch.Tensor,
        prompt_tokens: torch.Tensor,
        latent_tokens: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_token: Tensor [batch, hidden].
            prompt_tokens: Tensor [batch, prompt_len, hidden].
            latent_tokens: Optional tensor [batch, latent_len, hidden].
            prompt_mask: Optional bool tensor [batch, prompt_len].
            latent_mask: Optional bool tensor [batch, latent_len].
        """
        if visual_token.dim() != 2:
            raise ValueError("visual_token must have shape [batch, hidden].")
        if prompt_tokens.dim() != 3:
            raise ValueError("prompt_tokens must have shape [batch, prompt_len, hidden].")
        batch_size = int(visual_token.size(0))
        device = visual_token.device
        if latent_tokens is None:
            latent_tokens = visual_token.new_zeros((batch_size, 0, visual_token.size(-1)))
        if latent_tokens.dim() != 3:
            raise ValueError("latent_tokens must have shape [batch, latent_len, hidden].")
        if prompt_tokens.size(1) > self.max_prompt_tokens:
            prompt_tokens = prompt_tokens[:, -self.max_prompt_tokens :, :]
            if prompt_mask is not None:
                prompt_mask = prompt_mask[:, -self.max_prompt_tokens :]
        if latent_tokens.size(1) > self.max_latent_steps:
            latent_tokens = latent_tokens[:, -self.max_latent_steps :, :]
            if latent_mask is not None:
                latent_mask = latent_mask[:, -self.max_latent_steps :]

        cls = self.cls_token.view(1, 1, -1).expand(batch_size, 1, -1)
        visual = self.input_proj(visual_token).unsqueeze(1)
        prompt = self.input_proj(prompt_tokens)
        latent = self.input_proj(latent_tokens) if latent_tokens.size(1) else latent_tokens.new_zeros((batch_size, 0, self.controller_hidden_size))
        hidden = torch.cat([cls, visual, prompt, latent], dim=1)

        prompt_len = int(prompt.size(1))
        latent_len = int(latent.size(1))
        type_ids = torch.cat(
            [
                torch.full((batch_size, 1), self.TYPE_CLS, device=device, dtype=torch.long),
                torch.full((batch_size, 1), self.TYPE_IMAGE, device=device, dtype=torch.long),
                torch.full((batch_size, prompt_len), self.TYPE_PROMPT, device=device, dtype=torch.long),
                torch.full((batch_size, latent_len), self.TYPE_LATENT, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        pos_ids = torch.arange(hidden.size(1), device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        hidden = hidden + self.type_embedding(type_ids) + self.position_embedding(pos_ids)

        if prompt_mask is None:
            prompt_mask = torch.ones((batch_size, prompt_len), device=device, dtype=torch.bool)
        else:
            prompt_mask = prompt_mask.to(device=device, dtype=torch.bool)
        if latent_mask is None:
            latent_mask = torch.ones((batch_size, latent_len), device=device, dtype=torch.bool)
        else:
            latent_mask = latent_mask.to(device=device, dtype=torch.bool)
        valid_mask = torch.cat(
            [
                torch.ones((batch_size, 2), device=device, dtype=torch.bool),
                prompt_mask,
                latent_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        return self.stop_head(self.norm(encoded[:, 0, :])).squeeze(-1)

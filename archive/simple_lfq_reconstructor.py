import torch
import torch.nn as nn
from vector_quantize_pytorch import LFQ  # type: ignore


class SimpleReconstructor(nn.Module):
    def __init__(self, extern_vocab_size: int,
                 hidden_size: int = 128,
                 idea_token_vocab_size: int = 16):
        super().__init__()  # type: ignore
        self.quantizer = LFQ(
            dim=hidden_size,
            codebook_size=idea_token_vocab_size,
            has_projections=True,
            entropy_loss_weight=0.01,
            commitment_loss_weight=1.0,
            diversity_gamma=1.0,
            spherical=True,
        )
        extern_vocab_size = extern_vocab_size
        self.out_proj = nn.Linear(hidden_size, extern_vocab_size).to("cuda")
        self.embed = nn.Embedding(extern_vocab_size, hidden_size)

    def forward(self, tokens: torch.Tensor):
        embeddings = self.embed(tokens)
        # Get loss breakdown
        #    quantized, indices, aux_loss = self.quantizer(embeddings)
        ret, breakdown = self.quantizer(embeddings, return_loss_breakdown=True)
        # Only use codebook entropy for aux loss

        aux_loss = (
            breakdown.commitment
            - self.quantizer.diversity_gamma
            * breakdown.batch_entropy
            * self.quantizer.entropy_loss_weight
        )
        logits = self.out_proj(ret.quantized)
        return logits, ret.indices, aux_loss, breakdown

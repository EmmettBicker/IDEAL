import math
import random
from abc import abstractmethod

import torch
import torch.nn as nn
from vector_quantize_pytorch import LFQ  # type: ignore
from utils.tokenizer import ITokenizer


class ITranslator(nn.Module):
    @abstractmethod
    def __call__(self,
                 source_tokens: torch.Tensor,
                 padding_mask: torch.Tensor,
                 target_tokens: torch.Tensor,
                 tgt_padding_mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Abstract Method")


class IDEALTranslator(ITranslator, nn.Module):
    def __init__(
        self,
        tokenizer: ITokenizer,
        v2: bool = False,
        max_sequence_length: int = 512,
        vocab_size: int = 50257,
        idea_token_vocab_size: int = 1024,
        hidden_size: int = 768,
        dim_feedforward: int = 768,
        num_text_encoder_layers: int = 6,
        num_idea_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        no_per_sample_entropy: bool = True,
    ):
        super().__init__()  # type: ignore

        assert (
            idea_token_vocab_size & (idea_token_vocab_size - 1) == 0
        ) and idea_token_vocab_size != 0, \
            "idea_token_vocab_size must be a power of 2"

        # Model 1: GPT2 tokens to idea tokens
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_text_encoder_layers,
        )

        self.v2 = v2
        # v2: Adds an encoder after retokenization
        if self.v2:
            self.idea_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                ),
                num_layers=num_idea_encoder_layers,
            )

        self.idea_token_vocab_size = idea_token_vocab_size
        self.latent_dim = int(math.log2(self.idea_token_vocab_size))

        self.quantizer = LFQ(
            dim=hidden_size,
            codebook_size=idea_token_vocab_size,
            has_projections=True,
            num_codebooks=1,  # hyperparameter later
            entropy_loss_weight=0.01,  # hyperparameter later
            commitment_loss_weight=1,  # hyperparameter later
            diversity_gamma=2.5,  # hyperparameter later
            soft_clamp_input_value=10,  # hyperparameter later
            spherical=True,  # hyperparameter later
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )
        self.project_return = nn.Linear(hidden_size, vocab_size)

        self.tokenizer = tokenizer
        self.gpt2_embeddings = nn.Embedding(vocab_size + 1, hidden_size)  # bos
        self.pos_encoding = nn.Embedding(
            max_sequence_length + 1, hidden_size
        )  # Max sequence

        self.no_per_sample_entropy = no_per_sample_entropy

    def forward(
        self,
        source_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        tgt_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # B, L = source_tokens.size(0), source_tokens.size(1)

        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = (
            torch.arange(0, source_tokens.size(1), device=source_tokens.device)
            .unsqueeze(0)
            .expand_as(source_tokens)
        )
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings

        # ENCODER
        embeddings = self.encoder(
            src_embeddings,
            src_key_padding_mask=padding_mask
        )

        # LFQ
        idea_embeddings, aux_losses = self.get_idea_embeddings(embeddings)

        # Adds ENCODER 2 if v2 is specified
        if self.v2:
            # ENCODER 2
            idea_embeddings = self.idea_encoder(
                idea_embeddings, src_key_padding_mask=padding_mask
            )

        # Idea positional encodings
        tgt_embedded = self.gpt2_embeddings(target_tokens)
        position_indices = (
            torch.arange(0, tgt_embedded.size(1), device=source_tokens.device)
            .unsqueeze(0)
            .expand_as(target_tokens)
        )
        pos_embeddings = self.pos_encoding(position_indices)
        tgt_embedded = tgt_embedded + pos_embeddings

        # DECODER
        decoder_output = self.decoder(
            tgt_embedded,
            idea_embeddings,
            tgt_mask=self.generate_square_subsequent_mask(
                target_tokens.size(1)).to(tgt_embedded.device),
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        output_logits = self.project_return(decoder_output)

        if random.random() > 0.995:
            print(
                self.tokenizer.decode(  # type: ignore
                    source_tokens[0][~padding_mask[0]]),
                " | ",
                self.tokenizer.decode(  # type: ignore
                    output_logits[0].argmax(dim=-1)[~tgt_padding_mask[0]]
                ),
                " | ",
                self.tokenizer.decode(  # type: ignore
                    target_tokens[0][~tgt_padding_mask[0]]),
            )

        return output_logits, aux_losses

    def get_idea_embeddings(self,
                            embeddings: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.no_per_sample_entropy:
            ret, breakdown = self.quantizer(embeddings,
                                            return_loss_breakdown=True)
            aux_loss = (
                breakdown.commitment
                - self.quantizer.diversity_gamma
                * breakdown.batch_entropy
                * self.quantizer.entropy_loss_weight
            )
            quantized_output = ret.quantized
        else:
            quantized_output, _, aux_loss = self.quantizer(embeddings)

        return quantized_output, aux_loss

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask


class StandardTransformer(ITranslator, nn.Module):
    def __init__(
        self,
        tokenizer: ITokenizer,
        max_sequence_length: int = 512,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        dim_feedforward: int = 768,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):

        super().__init__()  # type: ignore

        # Model 1: GPT2 tokens to idea tokens
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        self.to_gpt2 = nn.Linear(hidden_size, vocab_size)

        self.tokenizer = tokenizer

        # Token embeddings
        self.gpt2_embeddings = nn.Embedding(vocab_size + 1, hidden_size)  # +1
        self.pos_encoding = nn.Embedding(
            max_sequence_length + 1, hidden_size
        )  # Max sequence length plus one for bos token
        # self.pos_encoding = nn.Embedding(513, hidden_size)

        self.tau = 0.5

    def forward(
        self,
        source_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        tgt_padding_mask: torch.Tensor
    ):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = (
            torch.arange(0, source_tokens.size(1), device=source_tokens.device)
            .unsqueeze(0)
            .expand_as(source_tokens)
        )
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings

        embeddings = self.encoder(
            src_embeddings,
            src_key_padding_mask=padding_mask
        )

        tgt_embedded = self.gpt2_embeddings(target_tokens)
        position_indices = (
            torch.arange(0, tgt_embedded.size(1), device=source_tokens.device)
            .unsqueeze(0)
            .expand_as(target_tokens)
        )
        pos_embeddings = self.pos_encoding(position_indices)

        tgt_embedded = tgt_embedded + pos_embeddings

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(
            target_tokens.size(1)).to(tgt_embedded.device)

        # Omit tgt_key_padding_mask
        decoder_output = self.decoder(
            tgt_embedded,
            embeddings,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        output_logits = self.to_gpt2(decoder_output)

        if random.random() > 0.995:
            print(
                self.tokenizer.decode(  # type: ignore
                    source_tokens[0][~padding_mask[0]]),
                " | ",
                self.tokenizer.decode(  # type: ignore
                    output_logits[0].argmax(dim=-1)[~tgt_padding_mask[0]]
                ),
                " | ",
                self.tokenizer.decode(  # type: ignore
                    target_tokens[0][~tgt_padding_mask[0]]),
            )

        return output_logits, None

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

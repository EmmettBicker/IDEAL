import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import math
import lfq


class IDEALTranslator(nn.Module):
    def __init__(
        self,
        tokenizer,
        max_sequence_length=512,
        vocab_size=50257,
        idea_token_vocab_size=1024,
        hidden_size=768,
        dim_feedforward=768,
        num_encoder_layers=6,
        num_decoder_layers=6,
        use_binary_latent_tokenization=True,
        hard=True

    ):

        super().__init__()

        # Model 1: GPT2 tokens to idea tokens
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        self.use_binary_latent_tokenization = use_binary_latent_tokenization
        self.idea_token_vocab_size = idea_token_vocab_size
        self.hard = hard
        
        if use_binary_latent_tokenization:
            assert (idea_token_vocab_size & (idea_token_vocab_size-1) == 0) and idea_token_vocab_size != 0, "If using binary latents, idea_token_vocab_size must be a power of 2"
            self.latent_dim = int(math.log2(self.idea_token_vocab_size))
            self.to_idea_binary_latents = nn.Linear(hidden_size, self.latent_dim) # Currently it's one to one
        else:
            self.to_idea_tokens = nn.Linear(hidden_size, idea_token_vocab_size) # Currently it's one to one
        

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        self.to_gpt2 = nn.Linear(hidden_size, vocab_size)

        self.tokenizer = tokenizer

        # Token embeddings
        self.gpt2_embeddings = nn.Embedding(vocab_size+1, hidden_size) # +1 for bos
        self.idea_embeddings = nn.Embedding(idea_token_vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(max_sequence_length+1, hidden_size) # Max sequence length plus one for bos token

        self.tau = 0.5
        
        
    def forward(self, source_tokens, padding_mask, target_tokens=None, tgt_padding_mask=None):
        B, L = source_tokens.size(0), source_tokens.size(1)
        
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.encoder(src_embeddings, src_key_padding_mask=padding_mask)
        

        if self.use_binary_latent_tokenization:
            
            idea_latents = self.to_idea_binary_latents(embeddings) # Becomes (batch x seq_len x log2(vocab_size) )                
            continous_probs = lfq.binary_latents_to_token_probs(idea_latents) # I'll have to disable this in eval at some point
            continous_embeddings = torch.matmul(continous_probs, self.idea_embeddings.weight)
            
            if self.hard:
                idea_tokens = lfq.binary_latents_to_hard_tokens(idea_latents, latent_dim=self.latent_dim)
                discrete_embeddings = self.idea_embeddings(idea_tokens)
                idea_embeddings = continous_embeddings + (discrete_embeddings - continous_embeddings).detach()
            else:
                idea_embeddings = continous_embeddings
        else:
            idea_logits = self.to_idea_tokens(embeddings)
            self.tau *= 0.999
            idea_tokens = F.gumbel_softmax(idea_logits, tau=self.tau, hard=True, dim=-1)
            idea_embeddings = torch.matmul(idea_tokens, self.idea_embeddings.weight)
            idea_embeddings = idea_embeddings + pos_embeddings

       
        tgt_embedded = self.gpt2_embeddings(target_tokens)
        position_indices = torch.arange(0, tgt_embedded.size(1), device=source_tokens.device).unsqueeze(0).expand_as(target_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        tgt_embedded = tgt_embedded + pos_embeddings

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(target_tokens.size(1)).to(tgt_embedded.device)

        # Omit tgt_key_padding_mask
        decoder_output = self.decoder(
            tgt_embedded,
            idea_embeddings,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
       
        output_logits = self.to_gpt2(decoder_output)

        if random.random() > 0.995:
          print(
                self.tokenizer.decode(source_tokens[0][~padding_mask[0]]), " | ",
                self.tokenizer.decode(output_logits[0].argmax(dim=-1)[~tgt_padding_mask[0]]), " | ",
                self.tokenizer.decode(target_tokens[0][~tgt_padding_mask[0]])
            )

        return output_logits, idea_tokens
    
    
        
    def _generate(self, idea_embeddings, padding_mask, max_length=50):
        # Start with a batch of the initial tokens (e.g., [BOS] token)
        batch_size = idea_embeddings.size(0)
        generated_tokens = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=idea_embeddings.device)
        
        for _ in range(max_length):
            tgt_embeddings = self.gpt2_embeddings(generated_tokens)
            position_indices = torch.arange(0, tgt_embeddings.size(1), device=generated_tokens.device).unsqueeze(0).expand_as(generated_tokens)
            pos_embeddings = self.pos_encoding(position_indices)

            tgt_embeddings = tgt_embeddings + pos_embeddings

            tgt_mask = self.generate_square_subsequent_mask(tgt_embeddings.size(1)).to(tgt_embeddings.device)

            decoder_output = self.decoder(
                tgt_embeddings,
                idea_embeddings,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=padding_mask
            )

            logits = self.to_gpt2(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated_tokens

    def generate(self, source_tokens, padding_mask):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.encoder(src_embeddings, src_key_padding_mask=padding_mask)
        idea_logits = self.to_idea_tokens(embeddings)

        self.tau *= 0.999
        idea_tokens = F.gumbel_softmax(idea_logits, tau=self.tau, hard=True, dim=-1)
        idea_embeddings = torch.matmul(idea_tokens, self.idea_embeddings.weight)
        idea_embeddings = idea_embeddings + pos_embeddings
        
        return self._generate(idea_embeddings, padding_mask)

    

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

class IDEALTranslatorV2(nn.Module):
    def __init__(
        self,
        tokenizer,
        max_sequence_length=512,
        vocab_size=50257,
        idea_token_vocab_size=1024,
        hidden_size=768,
        dim_feedforward=768,
        num_text_encoder_layers=3,
        num_idea_encoder_layers=3,
        num_decoder_layers=6,

    ):

        super().__init__()

        # Model 1: GPT2 tokens to idea tokens
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_text_encoder_layers
        )

        self.to_idea_tokens = nn.Linear(hidden_size, idea_token_vocab_size) # Currently it's one to one

        self.idea_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_idea_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        self.to_gpt2 = nn.Linear(hidden_size, vocab_size)

        self.tokenizer = tokenizer

        # Token embeddings
        self.gpt2_embeddings = nn.Embedding(vocab_size+1, hidden_size) # +1 for bos
        self.idea_embeddings = nn.Embedding(idea_token_vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(max_sequence_length+1, hidden_size) # Max sequence length plus one for bos token

        self.tau = 0.5
        
        
    def forward(self, source_tokens, padding_mask, target_tokens=None, tgt_padding_mask=None):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.text_encoder(src_embeddings, src_key_padding_mask=padding_mask)
        idea_logits = self.to_idea_tokens(embeddings)

        self.tau *= 0.999
        idea_tokens = F.gumbel_softmax(idea_logits, tau=self.tau, hard=True, dim=-1)
        idea_embeddings = torch.matmul(idea_tokens, self.idea_embeddings.weight)
        idea_embeddings = idea_embeddings + pos_embeddings
        
        idea_embeddings = self.idea_encoder(idea_embeddings, src_key_padding_mask=padding_mask)
       
        tgt_embedded = self.gpt2_embeddings(target_tokens)
        position_indices = torch.arange(0, tgt_embedded.size(1), device=source_tokens.device).unsqueeze(0).expand_as(target_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        tgt_embedded = tgt_embedded + pos_embeddings

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(target_tokens.size(1)).to(tgt_embedded.device)

        # Omit tgt_key_padding_mask
        decoder_output = self.decoder(
            tgt_embedded,
            idea_embeddings,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
       
        output_logits = self.to_gpt2(decoder_output)

        if random.random() > 0.995:
          print(
                self.tokenizer.decode(source_tokens[0][~padding_mask[0]]), " | ",
                self.tokenizer.decode(output_logits[0].argmax(dim=-1)[~tgt_padding_mask[0]]), " | ",
                self.tokenizer.decode(target_tokens[0][~tgt_padding_mask[0]])
            )

        return output_logits, idea_tokens
        
    def _generate(self, idea_embeddings, padding_mask, max_length=50):
        # Start with a batch of the initial tokens (e.g., [BOS] token)
        batch_size = idea_embeddings.size(0)
        generated_tokens = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=idea_embeddings.device)
        
        for _ in range(max_length):
            tgt_embeddings = self.gpt2_embeddings(generated_tokens)
            position_indices = torch.arange(0, tgt_embeddings.size(1), device=generated_tokens.device).unsqueeze(0).expand_as(generated_tokens)
            pos_embeddings = self.pos_encoding(position_indices)

            tgt_embeddings = tgt_embeddings + pos_embeddings

            tgt_mask = self.generate_square_subsequent_mask(tgt_embeddings.size(1)).to(tgt_embeddings.device)

            decoder_output = self.decoder(
                tgt_embeddings,
                idea_embeddings,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=padding_mask
            )

            logits = self.to_gpt2(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated_tokens

    def generate(self, source_tokens, padding_mask):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.text_encoder(src_embeddings, src_key_padding_mask=padding_mask)
        idea_logits = self.to_idea_tokens(embeddings)

        self.tau *= 0.999
        idea_tokens = F.gumbel_softmax(idea_logits, tau=self.tau, hard=True, dim=-1)
        idea_embeddings = torch.matmul(idea_tokens, self.idea_embeddings.weight)
        
        idea_embeddings = self.idea_encoder(idea_embeddings, src_key_padding_mask=padding_mask)
        idea_embeddings = idea_embeddings + pos_embeddings
        
        return self._generate(idea_embeddings, padding_mask)

    

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask


class StandardTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        max_sequence_length=512,
        vocab_size=50257,
        hidden_size=768,
        dim_feedforward=768,
        num_encoder_layers=6,
        num_decoder_layers=6,

    ):

        super().__init__()

        # Model 1: GPT2 tokens to idea tokens
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        self.to_gpt2 = nn.Linear(hidden_size, vocab_size)

        self.tokenizer = tokenizer

        # Token embeddings
        self.gpt2_embeddings = nn.Embedding(vocab_size+1, hidden_size) # +1 for bos
        self.pos_encoding = nn.Embedding(max_sequence_length+1, hidden_size) # Max sequence length plus one for bos token
        # self.pos_encoding = nn.Embedding(513, hidden_size)


        self.tau = 0.5

    def forward(self, source_tokens, padding_mask, target_tokens=None, tgt_padding_mask=None):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.encoder(src_embeddings, src_key_padding_mask=padding_mask)
       

        tgt_embedded = self.gpt2_embeddings(target_tokens)
        position_indices = torch.arange(0, tgt_embedded.size(1), device=source_tokens.device).unsqueeze(0).expand_as(target_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        tgt_embedded = tgt_embedded + pos_embeddings

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(target_tokens.size(1)).to(tgt_embedded.device)

        # Omit tgt_key_padding_mask
        decoder_output = self.decoder(
            tgt_embedded,
            embeddings,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        output_logits = self.to_gpt2(decoder_output)

        if random.random() > 0.995:
          print(
                self.tokenizer.decode(source_tokens[0][~padding_mask[0]]), " | ",
                self.tokenizer.decode(output_logits[0].argmax(dim=-1)[~tgt_padding_mask[0]]), " | ",
                self.tokenizer.decode(target_tokens[0][~tgt_padding_mask[0]])
            )

        return output_logits, None
    
    def generate(self, source_tokens, padding_mask):
        src_embeddings = self.gpt2_embeddings(source_tokens)
        position_indices = torch.arange(0, source_tokens.size(1), device=source_tokens.device).unsqueeze(0).expand_as(source_tokens)
        pos_embeddings = self.pos_encoding(position_indices)

        # Add positional encoding to source token embeddings
        src_embeddings = src_embeddings + pos_embeddings


        embeddings = self.encoder(src_embeddings, src_key_padding_mask=padding_mask)
        return self._generate(embeddings, padding_mask)

    def _generate(self, idea_embeddings, padding_mask, max_length=50):
        # Start with a batch of the initial tokens (e.g., [BOS] token)
        batch_size = idea_embeddings.size(0)
        generated_tokens = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=idea_embeddings.device)
        
        for _ in range(max_length):
            tgt_embeddings = self.gpt2_embeddings(generated_tokens)
            position_indices = torch.arange(0, tgt_embeddings.size(1), device=generated_tokens.device).unsqueeze(0).expand_as(generated_tokens)
            pos_embeddings = self.pos_encoding(position_indices)

            tgt_embeddings = tgt_embeddings + pos_embeddings

            tgt_mask = self.generate_square_subsequent_mask(tgt_embeddings.size(1)).to(tgt_embeddings.device)

            decoder_output = self.decoder(
                tgt_embeddings,
                idea_embeddings,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=padding_mask
            )

            logits = self.to_gpt2(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated_tokens
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

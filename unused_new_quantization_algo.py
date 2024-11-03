import torch
import torch.nn.functional as F

def binary_latents_to_hard_tokens(idea_latents, latent_dim):
    with torch.no_grad():
        hard_q = torch.sign(idea_latents)
        # Convert from {-1, 1} to {0, 1} for indexing
        binary_indices = (hard_q > 0).float()

        power_tensor = torch.pow(2, torch.arange(0, latent_dim, device=idea_latents.device)).flip(0)
        
        idea_tokens = torch.sum(binary_indices * power_tensor, dim=-1).long()
        return idea_tokens
    
def binary_latents_to_token_probs(idea_latents):
        digit_probs = F.sigmoid(idea_latents)
        # needs a loss term to push outputs closer to 0 and 1
        device = digit_probs.device
        B, L, num_digits = digit_probs.shape
        vocab_size = 2**num_digits
        
        
        indices_ones = torch.arange(vocab_size, device=device)[:, None].bitwise_and(
            2**torch.arange(num_digits-1, -1, -1, device=device)
        ).bool().float() 
        
        indices_zeros = 1 - torch.arange(vocab_size, device=device)[:, None].bitwise_and(
            2**torch.arange(num_digits-1, -1, -1, device=device)
        ).bool().float() 
        
        digit_probs = digit_probs.unsqueeze(2)
        a = indices_ones.expand(B,L, vocab_size,-1) * digit_probs
        b = indices_zeros * (1 - digit_probs)
        
        prob_of_every_digit = torch.prod(a+b,dim=-1)
        
        return prob_of_every_digit            
        
    

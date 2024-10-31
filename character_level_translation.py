# %%
try:
   from datasets import load_dataset
except:
   from datasets import load_dataset

from ideal_translator import IDEALTranslator, StandardTransformer

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

# Load both train and validation datasets
train_dataset = load_dataset("opus100", "en-fr", split="train[:90000]")
val_dataset = load_dataset("opus100", "en-fr", split="train[90000:100000]")

# Create character vocabulary
char_vocab = set()
for item in train_dataset:
   [char_vocab.add(x) for x in "".join(item['translation'].values())]
for item in val_dataset:
   [char_vocab.add(x) for x in "".join(item['translation'].values())]

# Add special tokens
special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
char_vocab = special_tokens + sorted(list(char_vocab))
char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class CharTokenizer:
   def __init__(self, char_to_idx):
       self.char_to_idx = char_to_idx
       self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
       self.pad_token_id = char_to_idx['<PAD>']
       self.bos_token_id = char_to_idx['<BOS>']
       self.eos_token_id = char_to_idx['<EOS>']
       self.unk_token_id = char_to_idx['<UNK>']
       
   def encode(self, text):
       return [self.char_to_idx.get(c, self.unk_token_id) for c in text]
       
   def decode(self, batched_ids):
        if isinstance(batched_ids, torch.Tensor) and batched_ids.ndim != 1:
            return [''.join([self.idx_to_char[id.item()] for id in ids if id not in [self.pad_token_id]]) for ids in batched_ids]
        else:
            return ''.join([self.idx_to_char[id.item()] for id in batched_ids if id not in [self.pad_token_id]])
       
   def __call__(self, texts, padding=True, return_tensors='pt', truncation=True, max_length=512):
       if isinstance(texts, str):
           texts = [texts]
           
       encoded = [self.encode(text) for text in texts]
       
       if truncation:
           encoded = [seq[:max_length] for seq in encoded]
           
       if padding:
           max_len = max(len(seq) for seq in encoded)
           attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in encoded]
           encoded = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in encoded]
           
       if return_tensors == 'pt':
           encoded = torch.tensor(encoded)
           attention_mask = torch.tensor(attention_mask)
           return type('Encoding', (), {'input_ids': encoded, 'attention_mask': attention_mask})
           
       return encoded

class TranslationDataset(Dataset):
   def __init__(self, dataset, tokenizer):
       self.dataset = dataset
       self.tokenizer = tokenizer

   def __len__(self):
       return len(self.dataset)

   def __getitem__(self, idx):
       item = self.dataset[idx]
       return {
           'source': item['translation']['fr'],
           'target': item['translation']['en']
       }

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = CharTokenizer(char_to_idx)

# Create both datasets
train_dataset = TranslationDataset(train_dataset, tokenizer)
val_dataset = TranslationDataset(val_dataset, tokenizer)

# Create both dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize models (assuming you kept the same model architectures but updated vocab size)
vocab_size = len(char_vocab)


models = {
    'IDEAL1': IDEALTranslator(tokenizer, idea_token_vocab_size=1, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),
    'IDEAL2': IDEALTranslator(tokenizer, idea_token_vocab_size=2, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),
    'IDEAL4': IDEALTranslator(tokenizer, idea_token_vocab_size=4, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),
    'IDEAL8': IDEALTranslator(tokenizer, idea_token_vocab_size=8, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),
    'IDEAL16': IDEALTranslator(tokenizer, idea_token_vocab_size=16,  vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),
    'IDEAL32': IDEALTranslator(tokenizer, idea_token_vocab_size=32, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device),

    'Standard': StandardTransformer(tokenizer, vocab_size=vocab_size, max_sequence_length=512, num_decoder_layers=1, num_encoder_layers=1, hidden_size=128).to(device)

    
    }

# Initialize optimizers in a dictionary
optimizers = {
    name: Adam(model.parameters(), lr=1e-4)
    for name, model in models.items()
}

# Initialize loss tracking
train_losses = {name: [] for name in models}
val_losses = {name: [] for name in models}
step_losses = {name: [] for name in models}


def validate(model, is_ideal=True):
   model.eval()
   total_loss = 0
   
   with torch.no_grad():
       for batch in val_loader:
           source_token_output = tokenizer(
               batch['source'],
               return_tensors='pt',
               padding=True,
               truncation=True,
               max_length=512,
           )

           target_tokens_output = tokenizer(
               batch['target'],
               return_tensors='pt',
               padding=True,
               truncation=True,
               max_length=512
           )

           source_tokens = source_token_output.input_ids.to(device)
           target_tokens = target_tokens_output.input_ids.to(device)

           bos = torch.tensor([[tokenizer.bos_token_id]]* target_tokens.size(0), device=target_tokens.device)
           target_tokens = torch.cat((bos, target_tokens), dim=1)
           eos = torch.tensor([[tokenizer.eos_token_id]]* target_tokens.size(0), device=target_tokens.device)
           shift_right_target_tokens = torch.cat((target_tokens[:,1:], eos), dim=1)

           padding_mask = ~source_token_output.attention_mask.bool().to(device)
           bos_bool = torch.tensor([[False]]* target_tokens.size(0), device=target_tokens.device)

           tgt_padding_mask = ~target_tokens_output.attention_mask.bool().to(device)
           tgt_padding_mask = torch.cat((bos_bool, tgt_padding_mask), dim=1)

           output_logits, _ = model(source_tokens, padding_mask, target_tokens, tgt_padding_mask)
           loss = F.cross_entropy(
                output_logits.view(-1, output_logits.size(-1)),
                shift_right_target_tokens.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
           )
            
           
           total_loss += loss.item()
           
   return total_loss / len(val_loader)

def train_epoch(model, optimizer, name):
   model.train()
   total_loss = 0
   
   for batch_idx, batch in enumerate(train_loader):
       optimizer.zero_grad()

       source_token_output = tokenizer(
           batch['source'],
           return_tensors='pt',
           padding=True,
           truncation=True,
           max_length=512,
       )

       target_tokens_output = tokenizer(
           batch['target'],
           return_tensors='pt',
           padding=True,
           truncation=True,
           max_length=512
       )

       source_tokens = source_token_output.input_ids.to(device)
       target_tokens = target_tokens_output.input_ids.to(device)

       bos = torch.tensor([[tokenizer.bos_token_id]]* target_tokens.size(0), device=target_tokens.device)
       target_tokens = torch.cat((bos, target_tokens), dim=1)
       eos = torch.tensor([[tokenizer.eos_token_id]]* target_tokens.size(0), device=target_tokens.device)
       shift_right_target_tokens = torch.cat((target_tokens[:,1:], eos), dim=1)

       padding_mask = ~source_token_output.attention_mask.bool().to(device)
       bos_bool = torch.tensor([[False]]* target_tokens.size(0), device=target_tokens.device)

       tgt_padding_mask = ~target_tokens_output.attention_mask.bool().to(device)
       tgt_padding_mask = torch.cat((bos_bool, tgt_padding_mask), dim=1)

       output_logits, idea_tokens = model(source_tokens, padding_mask, target_tokens, tgt_padding_mask)

       translation_loss = F.cross_entropy(
           output_logits.view(-1, output_logits.size(-1)),
           shift_right_target_tokens.view(-1),
           ignore_index=tokenizer.pad_token_id,
           reduction='mean'
       )

       total_loss += translation_loss.item()
       
       # Track loss every N batches
       if batch_idx % 50 == 0:
        step_losses[name].append(translation_loss.item())
        
       if batch_idx % 200 == 0:
        print(f'{name} Batch {batch_idx}, Loss: {translation_loss.item():.4f}')

           
       if batch_idx % 200 == 0:
           print(f'Batch {batch_idx}, Loss: {translation_loss.item():.4f}')

       translation_loss.backward()
       optimizer.step()

   return total_loss / len(train_loader)

def plot_losses():
    plt.figure(figsize=(12, 5))
    
    # Plot training loss over time (every N batches)
    plt.subplot(1, 2, 1)
    for name in models:
        plt.plot(step_losses[name], label=name)
    plt.title('Training Loss Over Steps')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot epoch-level losses
    plt.subplot(1, 2, 2)
    epochs = range(1, len(next(iter(train_losses.values()))) + 1)
    for name in models:
        plt.plot(epochs, train_losses[name], f'C{list(models.keys()).index(name)}-', label=f'{name} Train')
        plt.plot(epochs, val_losses[name], f'C{list(models.keys()).index(name)}--', label=f'{name} Val')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Training loop
num_epochs = 5
best_val_loss = float('inf')


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for name, model in models.items():
        # Train and validate
        train_loss = train_epoch(model, optimizers[name], name)
        val_loss = validate(model, name)
        
        train_losses[name].append(train_loss)
        val_losses[name].append(val_loss)
        
        print(f"{name} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot current progress
    plot_losses()
    
    # Save best model
    current_val_losses = {name: val_losses[name][-1] for name in models}
    min_val_loss = min(current_val_losses.values())
    
    if min_val_loss < best_val_loss:
        best_val_loss = min_val_loss
        checkpoint = {
            'epoch': epoch,
            'char_to_idx': char_to_idx,
        }
        for name in models:
            checkpoint.update({
                f'{name}_model': models[name].state_dict(),
                f'{name}_optimizer': optimizers[name].state_dict(),
                f'{name}_train_loss': train_losses[name][-1],
                f'{name}_val_loss': val_losses[name][-1],
            })
        torch.save(checkpoint, 'best_model.pt')



test_fr = "Bonjour, comment allez-vous?"
print("\nTest Translations:")
print(f"French: {test_fr}")
encoding = tokenizer(test_fr, return_tensors='pt')
for name, model in models.items():
    print(f"{name}: {tokenizer.decode(model.generate(encoding.input_ids.to(device), encoding.attention_mask.to(device).bool()))}")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    from datasets import load_dataset  # type: ignore # noqa

import typing

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, random_split

from src.model.ideal_translator import (IDEALTranslator, ITranslator)
from src.data.magvit_tokens_data import get_magvit_v2_dataset


vocab_size = 2**18
pad_token_id = vocab_size-1
dataset = get_magvit_v2_dataset(pad_token_id=pad_token_id)  # overlap
total_size = len(dataset)
train_size = int(0.9 * total_size)  # 90% for training
test_size = total_size - train_size  # 10% for testing# Setup

train_dataset, val_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create both dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

extra_tokens = 3

layers = 1
hidden_size = 18
num_heads = 1

models: dict[str, ITranslator] = {

    "IDEAL-2^18-1:1": IDEALTranslator(
        tokenizer=None,
        idea_token_vocab_size=2**18,
        vocab_size=vocab_size,
        use_binary_input_vocab_embed=True,
        idea_tokens_per_token=1,
        max_sequence_length=4096,
        num_decoder_layers=layers,
        num_text_encoder_layers=layers,
        hidden_size=hidden_size,
        num_heads=num_heads
    ).to(device),

    # "IDEAL-1024-1:1": IDEALTranslator(
    #     tokenizer,
    #     idea_token_vocab_size=1024,
    #     vocab_size=vocab_size,
    #     idea_tokens_per_token=1,
    #     max_sequence_length=512,
    #     num_decoder_layers=layers,
    #     num_text_encoder_layers=layers,
    #     hidden_size=hidden_size,
    # ).to(device),

    # "Standard": StandardTransformer(
    #     tokenizer=None,
    #     vocab_size=vocab_size + extra_tokens,
    #     max_sequence_length=4096,
    #     num_decoder_layers=layers,
    #     num_encoder_layers=layers,
    #     hidden_size=hidden_size,
    #     num_heads=num_heads
    # ).to(device),
}

# models['IDEAL-lfq1024'].load_state_dict(
#   torch.load("IDEAL-lfq1024.pth", weights_only=True))

# Initialize optimizers in a dictionary
optimizers = {name: Adam(model.parameters(), lr=1e-3) for name, model in
              models.items()}

# Initialize loss tracking
train_losses: dict[str, list[float]] = {name: [] for name in models}
val_losses: dict[str, list[float]] = {name: [] for name in models}
step_losses: dict[str, list[float]] = {name: [] for name in models}


def process_batch(model: ITranslator,
                  batch: tuple[torch.Tensor, torch.Tensor]):
    tensor, attn_mask = batch
    source_tokens = tensor.to(device)
    target_tokens = tensor.to(device)

    bos_token_id = 0
    eos_token_id = 1
    bos = torch.tensor([[bos_token_id]] * target_tokens.size(0),
                       device=device)
    target_tokens = torch.cat((bos, target_tokens), dim=1)

    # Add EOS to target
    eos = torch.tensor([[eos_token_id]] * source_tokens.size(0),
                       device=device)
    # implicit shift right because of the bos
    source_tokens = torch.cat((source_tokens, eos), dim=1)

    tgt_padding_mask = ~attn_mask.bool().to(device)
    padding_mask = ~attn_mask.bool().to(device)

    bos_bool = torch.tensor([[False]] * target_tokens.size(0),
                            device=source_tokens.device)
    tgt_padding_mask = torch.cat((bos_bool, tgt_padding_mask), dim=1)

    eos_bool = torch.tensor([[True]] * source_tokens.size(0),
                            device=source_tokens.device)
    padding_mask = torch.cat((padding_mask, eos_bool), dim=1)

    output_logits, aux_loss = model(
        source_tokens, padding_mask, target_tokens, tgt_padding_mask
    )

    def to_binary(tensor: torch.Tensor, num_bits: int = 18) -> torch.Tensor:
        tensor = tensor.to(torch.int32)
        binary = tensor.unsqueeze(-1).bitwise_and(
            2**torch.arange(num_bits, dtype=torch.int32).to(tensor.device)
            ).bool().float()
        return binary

    def masked_binary_cross_entropy(output_logits: torch.Tensor,
                                    source_tokens: torch.Tensor,
                                    pad_token: int,
                                    num_bits: int = 18) -> torch.Tensor:
        padding_mask = (source_tokens != pad_token).float()
        target_binary = to_binary(source_tokens, num_bits)

        expanded_mask = padding_mask.unsqueeze(-1).expand_as(target_binary)

        # Flatten all tensors
        output_flat = output_logits.reshape(-1)
        target_flat = target_binary.reshape(-1)
        mask_flat = expanded_mask.reshape(-1)

        # Compute binary cross entropy with logits
        elementwise_loss = F.binary_cross_entropy(
            torch.sigmoid(output_flat),
            target_flat,
            reduction='none'
        )

        # Apply mask and compute mean over non-padding elements
        masked_loss = elementwise_loss * mask_flat
        num_valid = mask_flat.sum()

        # Return average loss over non-padding elements
        return masked_loss.sum() / (num_valid + 1e-6)

    loss = masked_binary_cross_entropy(
        output_logits,
        source_tokens,
        pad_token=pad_token_id
    )

    return loss, aux_loss


def validate(model: ITranslator):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:

            loss, _ = process_batch(model, batch)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_epoch(model: ITranslator, optimizer: Optimizer, name: str):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        translation_loss, aux_loss = process_batch(model, batch)
        og_translation_loss = translation_loss.detach().clone()

        # Track loss every N batches
        if batch_idx % 50 == 0:
            step_losses[name].append(translation_loss.item())

        if name != "Standard":
            translation_loss += aux_loss

        total_loss += translation_loss.item()

        if batch_idx % 200 == 0:
            print(
                f"{name} Batch {batch_idx}, Total Loss: {translation_loss.item():.4f}, Loss: {og_translation_loss.item():.4f}" # noqa
            )
            if name != "Standard":
                print(f"Aux loss: {aux_loss}")

        translation_loss.backward()  # type: ignore
        optimizer.step()

    return total_loss / len(train_loader)


@typing.no_type_check
def plot_losses():
    plt.figure(figsize=(12, 5))

    # Plot training loss over time (every N batches)
    plt.subplot(1, 2, 1)
    for name in models:
        plt.plot(step_losses[name], label=name)
    plt.title("Training Loss Over Steps")
    plt.xlabel("Steps (x50)")
    plt.ylabel("Loss")
    plt.legend()

    # Plot epoch-level losses
    plt.subplot(1, 2, 2)
    epochs = range(1, len(next(iter(train_losses.values()))) + 1)
    for name in models:
        plt.plot(
            epochs,
            train_losses[name],
            f"C{list(models.keys()).index(name)}-",
            label=f"{name} Train",
        )
        plt.plot(
            epochs,
            val_losses[name],
            f"C{list(models.keys()).index(name)}--",
            label=f"{name} Val",
        )
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Training loop
    num_epochs = 70
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for name, model in models.items():
            # Train and validate
            train_loss = train_epoch(model, optimizers[name], name)
            val_loss = validate(model)

            train_losses[name].append(train_loss)
            val_losses[name].append(val_loss)

            print(f"{name} Train Loss: {train_loss:.4f}, Val Loss: { val_loss:.4f}") # noqa

        # Plot current progress
        plot_losses()

        # Save best model
        current_val_losses = {name: val_losses[name][-1] for name in models}
        min_val_loss = min(current_val_losses.values())

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            checkpoint: dict[str, typing.Any] = {  # type: ignore
                "epoch": epoch
            }
            for name in models:
                checkpoint.update(  # type: ignore
                    {
                        f"{name}_model": models[name].state_dict(),
                        f"{name}_optimizer": optimizers[name].state_dict(),
                        f"{name}_train_loss": train_losses[name][-1],
                        f"{name}_val_loss": val_losses[name][-1],
                    }
                )
            torch.save(checkpoint, "best_model.pt")  # type: ignore

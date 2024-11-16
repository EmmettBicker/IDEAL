try:
    from datasets import load_dataset  # type: ignore
except Exception:
    from datasets import load_dataset  # type: ignore

import typing
from typing import Dict, TypedDict, cast
from utils.tokenizer import CharTokenizer, GPT2Tokenizer, Encoding

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from src.model.ideal_translator import (IDEALTranslator, ITranslator,
                                        StandardTransformer)


class TranslationPair(TypedDict):
    en: str
    fr: str


class DataItem(TypedDict):
    id: str
    translation: TranslationPair


# Load both train and validation datasets
base_train_dataset: Dataset[DataItem] = load_dataset(
    "tatoeba", lang1="en", lang2="fr", split="train[:9000]",
    trust_remote_code=True
)  # type: ignore
base_val_dataset: Dataset[DataItem] = load_dataset(
    "tatoeba", lang1="en", lang2="fr", split="train[9000:10000]",
    trust_remote_code=True
)  # type: ignore

# Create character vocabulary
set_char_vocab: set[str] = set()
for item in base_train_dataset:
    values = cast(list[str], list(item["translation"].values()))
    [set_char_vocab.add(x) for x in "".join(values)]
for item in base_val_dataset:
    values = cast(list[str], list(item["translation"].values()))
    [set_char_vocab.add(x) for x in "".join(values)]

# Add special tokens
special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

char_vocab: list[str] = special_tokens + sorted(list(set_char_vocab))
char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}


class TranslationDataset(Dataset[Dict[str, str]]):
    def __init__(self, dataset: Dataset[DataItem]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        return {
            "source": item["translation"]["fr"],
            "target": item["translation"]["en"],
        }


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CharTokenizer(char_to_idx)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.add_special_tokens(
#   {'pad_token': '<|PAD|>', 'bos_token': '<|BOS|>'})


# Create both datasets
train_dataset = TranslationDataset(base_train_dataset)
val_dataset = TranslationDataset(base_val_dataset)

# Create both dataloaders
train_loader = DataLoader(train_dataset, batch_size=480, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=480, shuffle=False)

if isinstance(tokenizer, GPT2Tokenizer):
    vocab_size = len(tokenizer)
else:
    vocab_size = len(char_vocab)

layers = 1
hidden_size = 128
models: dict[str, ITranslator] = {

    "IDEAL-lfq1024": IDEALTranslator(
        tokenizer,
        idea_token_vocab_size=1024,
        vocab_size=vocab_size,
        max_sequence_length=512,
        num_decoder_layers=layers,
        num_text_encoder_layers=layers,
        hidden_size=hidden_size,
    ).to(device),

    "Standard": StandardTransformer(
        tokenizer,
        vocab_size=vocab_size,
        max_sequence_length=512,
        num_decoder_layers=layers,
        num_encoder_layers=layers,
        hidden_size=hidden_size,
    ).to(device),
}

# models['IDEAL-lfq1024'].load_state_dict(
#   torch.load("IDEAL-lfq1024.pth", weights_only=True))

# Initialize optimizers in a dictionary
optimizers = {name: Adam(model.parameters(), lr=1e-4) for name, model in
              models.items()}

# Initialize loss tracking
train_losses: dict[str, list[float]] = {name: [] for name in models}
val_losses: dict[str, list[float]] = {name: [] for name in models}
step_losses: dict[str, list[float]] = {name: [] for name in models}


def validate(model: ITranslator):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            source_token_output = tokenizer(
                batch["source"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # target_tokens_output = tokenizer(
            #     batch["target"],
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=512,
            # )
            source_token_output = cast(Encoding, source_token_output)
            source_tokens: torch.Tensor = source_token_output.input_ids.to(
                device)
            target_tokens: torch.Tensor = source_token_output.input_ids.to(
                device)  # CHANGE BACK TO TARGET

            bos = torch.tensor(
                [[tokenizer.bos_token_id]] * target_tokens.size(0),
                device=target_tokens.device,
            )
            target_tokens = torch.cat((bos, target_tokens), dim=1)
            eos = torch.tensor(
                [[tokenizer.eos_token_id]] * target_tokens.size(0),
                device=target_tokens.device,
            )
            shift_right_target_tokens = torch.cat((target_tokens[:, 1:], eos),
                                                  dim=1)

            padding_mask = ~source_token_output.attention_mask.bool().to(
                device)
            bos_bool = torch.tensor(
                [[False]] * target_tokens.size(0), device=target_tokens.device
            )

            tgt_padding_mask = ~source_token_output.attention_mask.bool().to(
                device
            )  # CHANGE BACK TO TARGET
            tgt_padding_mask = torch.cat((bos_bool, tgt_padding_mask), dim=1)

            output_logits, _ = model(
                source_tokens, padding_mask, target_tokens, tgt_padding_mask
            )

            loss = F.cross_entropy(
                output_logits.view(-1, output_logits.size(-1)),
                shift_right_target_tokens.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="mean",
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_epoch(model: ITranslator, optimizer: Optimizer, name: str):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        source_token_output = tokenizer(
            batch["source"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # target_tokens_output = tokenizer(
        #     batch["target"],
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        # )

        source_token_output = cast(Encoding, source_token_output)
        source_tokens = source_token_output.input_ids.to(device)
        target_tokens = source_token_output.input_ids.to(
            device
        )  # CHANGE BACK TO TARGET

        bos = torch.tensor(
            [[tokenizer.bos_token_id]] * target_tokens.size(0),
            device=target_tokens.device,
        )
        target_tokens = torch.cat((bos, target_tokens), dim=1)
        eos = torch.tensor(
            [[tokenizer.eos_token_id]] * target_tokens.size(0),
            device=target_tokens.device,
        )
        shift_right_target_tokens = torch.cat(
            (target_tokens[:, 1:], eos), dim=1
        )

        padding_mask = ~source_token_output.attention_mask.bool().to(device)
        bos_bool = torch.tensor(
            [[False]] * target_tokens.size(0), device=target_tokens.device
        )

        tgt_padding_mask = ~source_token_output.attention_mask.bool().to(
            device
        )  # CHANGE BACK TO TARGET
        tgt_padding_mask = torch.cat((bos_bool, tgt_padding_mask), dim=1)

        output_logits, aux_loss = model(
            source_tokens, padding_mask, target_tokens, tgt_padding_mask
        )

        translation_loss = F.cross_entropy(
            output_logits.view(-1, output_logits.size(-1)),
            shift_right_target_tokens.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="mean",
        )

        og_translation_loss = translation_loss.detach().clone()

        # Track loss every N batches
        if batch_idx % 50 == 0:
            step_losses[name].append(translation_loss.item())

        if name != "Standard":
            translation_loss += aux_loss

        total_loss += translation_loss.item()

        if batch_idx % 200 == 0:
            print(
                f"{name} Batch {batch_idx}, Total Loss: {
                    translation_loss.item():.4f}, Loss: {
                        og_translation_loss.item():.4f}"
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

            print(f"{name} Train Loss: {train_loss:.4f}, Val Loss: {
                val_loss:.4f}")

        # Plot current progress
        plot_losses()

        # Save best model
        current_val_losses = {name: val_losses[name][-1] for name in models}
        min_val_loss = min(current_val_losses.values())

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            checkpoint = {  # type: ignore
                "epoch": epoch,
                "char_to_idx": char_to_idx,
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

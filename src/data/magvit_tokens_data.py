import torch
from torch.utils.data import Dataset
from io import BytesIO
from zipfile import ZipFile
import shutil
import os


def concatenate_and_maxpad(
        tensors: list[torch.Tensor],
        pad_value: int = -1
        ) -> torch.Tensor:
    # Find the maximum length of the tensors. Assumes input is batch x cat_dim
    max_length = max(tensor.size(1) for tensor in tensors)

    padded_tensors = [torch.nn.functional.pad(
        tensor, (0, max_length - tensor.size(1)),
        value=pad_value) for tensor in tensors]

    return torch.cat(padded_tensors, dim=0)


class MagvitV2TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tensor: torch.Tensor, pad_token_id: int):
        """
        Args:
            tensor: The tensor of shape [49345, 3969]
        """

        self.attn_mask = tensor != -1
        self.data = torch.where(tensor == -1, pad_token_id, tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.attn_mask[idx]


def get_magvit_v2_dataset(pad_token_id: int) -> MagvitV2TokenDataset:
    if not os.path.exists('src/data/image_tokens.zip'):
        shutil.make_archive('src/data/image_tokens',
                            'zip',
                            'src/data/image_tokens/')
    with ZipFile('src/data/image_tokens.zip', 'r') as zipf:
        # List all files
        files = zipf.namelist()
        files = [file for file in files if file[-3:] == ".pt"]
        max_lengths: list[int] = []
        tensors: list[torch.Tensor] = []
        for torch_file in files:
            torch_bytes = zipf.read(torch_file)

            torch_buffer = BytesIO(torch_bytes)

            tensor: torch.Tensor = torch.load(torch_buffer,  # type: ignore
                                              weights_only=True)
            tensors.append(tensor.squeeze())
            max_lengths.append(tensor.size(2))
        full_tensor = concatenate_and_maxpad(tensors)
        dataset = MagvitV2TokenDataset(full_tensor, pad_token_id)
        return dataset

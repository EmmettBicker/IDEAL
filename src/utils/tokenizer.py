from abc import abstractmethod
from typing import Union, cast

import torch
from transformers import GPT2Tokenizer as HFGPT2Tokenizer  # type: ignore


class ITokenizer():
    @abstractmethod
    def decode(
        self,
        token_ids: Union[int,  # type: ignore
                         list[int],
                         "np.ndarray",  # type: ignore # noqa: F821
                         "torch.Tensor",
                         "tf.Tensor"],  # type: ignore # noqa: F821
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,  # type: ignore
        **kwargs,  # type: ignore
    ) -> str:
        raise NotImplementedError("Abstract method")


class Encoding:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class CharTokenizer(ITokenizer):
    def __init__(self, char_to_idx: dict[str, int]):
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        self.pad_token_id = char_to_idx["<PAD>"]
        self.bos_token_id = char_to_idx["<BOS>"]
        self.eos_token_id = char_to_idx["<EOS>"]
        self.unk_token_id = char_to_idx["<UNK>"]

    def encode(self, text: str):
        return [self.char_to_idx.get(c, self.unk_token_id) for c in text]

    def decode(self,
               token_ids: Union[int,  # type: ignore
                                list[int],
                                "np.ndarray",  # type: ignore # noqa: F821
                                "torch.Tensor",
                                "tf.Tensor"],  # type: ignore # noqa: F821
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = None,  # type: ignore
               **kwargs,  # type: ignore
               ) -> str:
        token_ids: torch.Tensor = cast(torch.Tensor, token_ids)
        return "".join(
            [
                self.idx_to_char[int(id.item())]
                for id in token_ids
                if id.item() not in [self.pad_token_id]
            ]
        )

    def __call__(
        self,
        texts: str | list[str],
        padding: bool = True,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self.encode(text) for text in texts]

        if truncation:
            encoded = [seq[:max_length] for seq in encoded]

        attention_mask = None
        if padding:
            max_len = max(len(seq) for seq in encoded)
            attention_mask = [
                [1] * len(seq) + [0] * (max_len - len(seq)) for seq in encoded
            ]
            encoded = [
                seq +
                [self.pad_token_id] * (max_len - len(seq)) for seq in encoded
            ]

        if return_tensors == "pt":
            encoded = torch.tensor(encoded)
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask)
                return Encoding(
                    input_ids=encoded,
                    attention_mask=attention_mask
                )

            else:
                raise NotImplementedError("Haven't implemented for \
                                           padding=false")

        return encoded


class GPT2Tokenizer(ITokenizer, HFGPT2Tokenizer):
    pass

from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import sentencepiece
import tiktoken
import torch
from torch.utils.data import random_split, TensorDataset, Subset


def split_dataset(dataset: TensorDataset) -> List[Subset]:
    """
    Partition the input data into a training and the test set
    10% of the training set, or up to 1000 examples
    """
    n = min(1000, int(len(dataset) * 0.1))
    return random_split(dataset, [len(dataset) - n, n])


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, ints: torch.Tensor) -> str:
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        ...


class CharTokenizer(Tokenizer):
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        print(f'CharTokenizer vocabulary: {"".join(self.chars)}')
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[s] for s in text], dtype=torch.long)

    def decode(self, data: torch.Tensor) -> str:
        return "".join(self.itos[i] for i in data.tolist())

    def vocab_size(self) -> int:
        return len(self.chars)


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding_name="gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.enc.encode_ordinary(text), dtype=torch.long)

    def decode(self, data: torch.Tensor) -> str:
        return self.enc.decode(data.tolist())

    def vocab_size(self) -> int:
        return self.enc.n_vocab


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, text_path: Path):
        prefix = str(text_path.with_suffix("")) + "-sp"
        sentencepiece.SentencePieceTrainer.Train(input=text_path, model_prefix=prefix)
        self.enc = sentencepiece.SentencePieceProcessor(model_file=prefix + ".model")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.enc.encode(text), dtype=torch.long)

    def decode(self, data: torch.Tensor) -> str:
        return "".join(self.enc.decode(data.tolist()))

    def vocab_size(self) -> int:
        return self.enc.vocab_size()


class TransformerDataset(TensorDataset):
    def __init__(self, data: torch.Tensor, context_len: int):
        super().__init__()
        self.context_len = context_len
        self.data = data

        self.train = None
        self.test = None

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.context_len]
        y = self.data[index + 1 : index + self.context_len + 1]
        return x, y

    def __len__(self):
        return len(self.data) - self.context_len - 1

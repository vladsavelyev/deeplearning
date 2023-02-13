from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, Subset
import tiktoken


def create_dataset(path: Path, dataset_class: str, block_size: int) -> "MyDataset":
    pickled_path = path.with_suffix(".pt")
    if pickled_path.exists():
        print(f"Loading pickled dataset from {pickled_path}...")
        dataset = torch.load(pickled_path)
        return dataset

    print(f"Creating dataset from file {path}...")
    with path.open("r") as f:
        text: str = f.read()
    if dataset_class == WordListDataset.__name__:
        dataset = WordListDataset(text.splitlines())
    elif dataset_class == TextDataset.__name__:
        dataset = TextDataset(text, block_size=block_size, tokenize=char_tokenize)
    else:
        raise ValueError(f"Unknown dataset type {dataset_class}")

    print(f"Saving dataset to {pickled_path}...")
    torch.save(dataset, pickled_path)
    return dataset


class MyDataset(Dataset, ABC):
    """
    Adding a few methods to the Dataset class
    """

    @abstractmethod
    def __init__(self):
        self.stoi: dict[str, int] = NotImplemented
        self.itos: dict[int, str] = NotImplemented
        self.x = NotImplemented
        self.y = NotImplemented
        self.block_size: int = NotImplemented
        self.vocab_size: int = NotImplemented
        self.train_set: Subset = NotImplemented
        self.test_set: Subset = NotImplemented

    def encode(self, word) -> torch.Tensor:
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ints: list[int]) -> str:
        return "".join(self.itos[i] for i in ints)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    @abstractmethod
    def sample_and_print(
        self,
        model,
        device: torch.device,
        top_k: int | None = None,
        clean: bool = False,
        **kwargs,
    ):
        ...


def word_tokenize(text: str) -> dict[str, int]:
    words = sorted(list(set(text.split())))
    return {s: i for i, s in enumerate(words)}


def char_tokenize(text: str) -> dict[str, int]:
    chars = sorted(list(set(text)))
    return {s: i for i, s in enumerate(chars)}


def tiktoken_tokenize(text: str) -> dict[str, int]:
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    print(f"Dataset {len(ids):,} tokens")
    return {enc.decode([id]): id for id in ids}


class TextDataset(MyDataset):
    def __init__(
        self, text: str, block_size: int, tokenize: Callable[[str], dict[str, int]]
    ):
        super().__init__()
        self.block_size = block_size
        self.stoi = tokenize(text)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        print(f"Text size: {len(text)}")
        print(f"Block size: {self.block_size}")
        print(f"Number of unique tokens in the vocabulary: {self.vocab_size}")

        print("Converting the text into tensors...")
        self.x = torch.zeros((len(text) - block_size, block_size), dtype=torch.long)
        self.y = torch.zeros((len(text) - block_size, block_size), dtype=torch.long)
        for token_idx in range(len(text) - block_size):
            wx = text[token_idx : token_idx + block_size]
            wy = text[token_idx + 1 : token_idx + block_size + 1]
            self.x[token_idx, :] = self.encode(wx.split())
            self.y[token_idx, :] = self.encode(wy.split())

        print("Splitting into train and test...")
        # Partition the input data into a training and the test set
        # 10% of the training set, or up to 1000 examples
        test_set_size: int = min(1000, int(len(self.x) * 0.1))
        self.train_set, self.test_set = torch.utils.data.random_split(
            self, [len(self.x) - test_set_size, test_set_size]
        )
        print(
            f"Split the dataset into {len(self.train_set)} training examples "
            f"and {len(self.test_set)} test examples"
        )

    def sample_and_print(
        self,
        model,
        device: torch.device,
        top_k: int | None = None,
        clean: bool = False,
        max_new_tokens: int = 500,
        prompt: str = "\n",
    ):
        """
        Samples from the model and pretty prints the decoded samples
        """
        x_init = self.encode(prompt).to(device).view(1, -1)

        x_sampled = model.generate(
            x_init, max_new_tokens=max_new_tokens, top_k=top_k, do_sample=True
        ).to("cpu")
        x_sampled = x_sampled[:, len(prompt) :]  # remove the prompt

        # Get the sample_i'th row of sampled integers, as a Python list
        row: list[int] = x_sampled[0].tolist()
        print(self.decode(row))


class WordListDataset(MyDataset):
    """
    List of words, tokenized on the character level
    """

    def __init__(self, words: list[str]):
        words = self._clean_words(words)
        chars = sorted(list(set("".join(words))))
        self.vocab_size = len(chars) + 1  # characters followed by special 0 token

        max_word_length: int = max(len(w) for w in words)
        self.block_size = 1 + max_word_length  # <START> token followed by characters

        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.itos[0] = "."

        print(f"Number of examples in the dataset: {len(words)}")
        print(f"Max word length: {max_word_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:", "".join(chars))

        self.x = torch.zeros((len(words), self.block_size), dtype=torch.long)
        self.y = torch.zeros((len(words), self.block_size), dtype=torch.long)
        for i, word in enumerate(words):
            t = self.encode(word)
            self.x[i, 1 : 1 + len(t)] = t
            self.y[i, :-1] = self.x[i, 1:]
            # Index -1 will mask the loss at the inactive locations. We don't train
            # the model with the index -1: remember the embedding matrix takes x as
            # indices,
            # so index -1 wouldn't even make sense; so our vocabulary has only the "0"
            # <START> character along with real characters encoded as 1-27. Our "y" will
            # only be used in the cross_entropy function, where we explicitly pass
            # ignore_index=-1, so it doesn't look at these values in the "y" tensor.
            # When generating words, we use the prompt that wouldn't contain -1
            # characters.
            self.y[i, len(t) + 1 :] = -1

        # Partition the input data into a training and the test set
        # 10% of the training set, or up to 1000 examples
        test_set_size: int = min(1000, int(len(words) * 0.1))
        self.train_set, self.test_set = torch.utils.data.random_split(
            self, [len(words) - test_set_size, test_set_size]
        )
        print(
            f"Split the dataset into {len(self.train_set)} training examples "
            f"and {len(self.test_set)} test examples"
        )

    @staticmethod
    def _clean_words(words: list[str]) -> list[str]:
        words = [w.strip() for w in words]  # Get rid of any leading/trailing space
        words = [w for w in words if w]  # Get rid of any empty strings
        return words

    def sample_and_print(
        self,
        model,
        device: torch.device,
        top_k: int | None = None,
        clean: bool = True,
        num: int = 10,
    ):
        """
        Samples from the model and pretty prints the decoded samples
        """
        x_init = torch.zeros(num, 1, dtype=torch.long).to(device)

        # -1 because we already start with <START> token (index 0)
        n_steps = self.block_size - 1

        x_sampled = model.generate(
            x_init, max_new_tokens=n_steps, top_k=top_k, do_sample=True
        ).to("cpu")
        if clean:
            x_sampled = x_sampled[:, 1:]  # remove the "0" <START> token

        train_samples, test_samples, new_samples = [], [], []

        for sample_i in range(x_sampled.shape[0]):
            # Get the sample_i'th row of sampled integers, as a Python list
            row: list[int] = x_sampled[sample_i].tolist()

            if clean:
                # Token "0" is also the <STOP> token, so we crop the output sequence
                # at that point
                crop_from = row.index(0) if 0 in row else len(row)
                row = row[:crop_from]

            word_sample = self.decode(row)

            # separately track samples that we have and have not seen before
            if word_sample in self.train_set:
                train_samples.append(word_sample)
            elif word_sample in self.test_set:
                test_samples.append(word_sample)
            else:
                new_samples.append(word_sample)

        for samples, desc in [
            (train_samples, "in train"),
            (test_samples, "in test"),
            (new_samples, "new"),
        ]:
            print(f"{len(samples)} samples that are {desc}:")
            print("\n".join(samples))


DATASET_CLASSES = [TextDataset, WordListDataset]

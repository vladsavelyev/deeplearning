from typing import Optional, List
from abc import abstractmethod, ABC
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import random_split, Dataset

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer, BaseTokenizer



@dataclass
class InputDataClass:
    x: torch.Tensor
    y: Optional[torch.Tensor]


class TransformerDataset(Dataset, ABC):
    def __init__(
        self,
        text_path: Path,
    ):
        super().__init__()
        self.path = text_path
        
        self.tokenizer = self.load_or_create_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.load_or_populate_data(text_path)

        test_n = min(1000, int(len(self) * 0.1))
        self.train, self.test = random_split(self, [len(self) - test_n, test_n])

    @abstractmethod
    def get_context_len(self) -> int:
        ...

    def get_ignore_index(self) -> int:
        return -1

    @abstractmethod
    def create_tokenizer(self) -> BaseTokenizer:
        ...

    def load_or_create_tokenizer(self) -> BaseTokenizer:
        save_path = self.path.with_name(self.path.stem + "-tokenizer.json")
        if save_path.exists():
            print(f"Loading tokenizer from {save_path}")
            return BaseTokenizer(Tokenizer.from_file(str(save_path)))
        
        tokenizer = self.create_tokenizer()
        print(f"Created tokenizer with vocab size of {tokenizer.get_vocab_size()}")
        tokenizer.save(str(save_path))
        print(f"Saved tokenizer to {save_path}")
        if tokenizer.get_vocab_size() < 100:
            t = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
            t = {k: v for k, v in t}
            print(f"Tokenizer vocab: {t}")
        return tokenizer

    @abstractmethod
    def load_or_populate_data(self, text_path: Path):
        ...

    @abstractmethod
    def __getitem__(self, index) -> InputDataClass:
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def print_examples(self, model: torch.nn.Module, **kwargs):
        ...


class MurakamiDataset(TransformerDataset):
    train_vocab_size = 30_000
    source = "murakami/murakami.txt"

    def __init__(self, base_path: Path):
        self.data = None
        super().__init__(base_path / self.source)

    def get_context_len(self) -> int:
        return 32

    def load_or_populate_data(self, text_path: Path):
        if (pickle_path := text_path.with_suffix(".pt")).exists():
            print(f"Loading dataset from {pickle_path}")
            self.data = torch.load(pickle_path)
            return

        with text_path.open() as f:
            text = f.read()
        self.data = torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)
        print(f"Saving dataset to {pickle_path}")
        torch.save(self.data, pickle_path)

    def __getitem__(self, index) -> InputDataClass:
        x = self.data[index : index + self.get_context_len()]
        y = self.data[index + 1 : index + self.get_context_len() + 1]
        return InputDataClass(x, y)

    def __len__(self):
        return len(self.data) - self.get_context_len() - 1

    def create_tokenizer(self):
        tokenizer = ByteLevelBPETokenizer()
        max_chars = 100_000
        print(f"Training tokenizer from {self.path}")
        with open(self.path, "r") as f:
            text = f.read()
            if max_chars:
                print(f"Limiting to {max_chars} characters")
                text = text[:max_chars]
            tokenizer.train_from_iterator([text], vocab_size=self.train_vocab_size)
        return tokenizer

    def print_examples(
        self,
        model: torch.nn.Module,
        **kwargs,
    ):
        prompt = "Охота на овец\nЧасть первая"
        prompt = "\n" * (self.get_context_len()) + prompt  # padding the prompt
        print(f"Prompt length is {len(prompt)}")
        prompt = torch.tensor(self.tokenizer.encode(prompt).ids)
        x_sampled = model.generate(prompt, **kwargs).to("cpu")
        text = self.tokenizer.decode(x_sampled[0].tolist(), skip_special_tokens=False)
        print(text)


class CharTokenizer:
    ignore_index = -1

    def __init__(self, text: str):
        chars = sorted(set("".join(text.split())))
        self.vocab_size = len(chars) + 1
        self.itos = {i + 1: c for i, c in enumerate(chars)}
        self.itos[0] = '.'
        self.itos[self.ignore_index] = '_'
        self.stoi = {c: i for i, c in self.itos.items()}

    def get_vocab(self):
        return self.itos
    
    def token_to_id(self, token: str) -> int:
        return self.stoi[token]
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i != 0]
        return "".join(self.itos[i] for i in ids)
    

class NamesDataset(TransformerDataset):
    source = "names/names.txt"

    def __init__(self, base_path: Path):
        self.x = None
        self.y = None
        self.context_len = None
        super().__init__(base_path / self.source)

    def get_context_len(self) -> int:
        return self.context_len

    def get_ignore_index(self) -> int:
        return CharTokenizer.ignore_index
    
    def __getitem__(self, index) -> InputDataClass:
        assert len(self.x[index]) == self.context_len, len(self.x[index])
        assert len(self.y[index]) == self.context_len, len(self.y[index])
        return InputDataClass(self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

    def load_or_create_tokenizer(self) -> CharTokenizer:
        return self.create_tokenizer()

    def create_tokenizer(self) -> CharTokenizer:
        with self.path.open() as f:
            text = f.read()
        return CharTokenizer(text)

    def load_or_populate_data(self, text_path: Path):
        with text_path.open() as f:
            names = f.read().split()
        max_name_len = max(len(n) for n in names)
        self.context_len = max_name_len + 1
        tokenizer = self.tokenizer

        xs = []
        ys = []
        for name in names:
            # name = "." + name + "."
            # """
            # ....... -> _____.e
            # ......e -> ____.em
            # .....em -> ___.emm
            # ....emm -> __.emma
            # ...emma -> _.emma.
            # ....... -> _____.o
            # ......o -> ____.ol
            # .....ol -> ___.oli
            # ....oli -> __.oliv
            # ...oliv -> _.olivi
            # ..olivi -> .olivia
            # .olivia -> olivia.
            # """
            for i in range(1, len(name)):
                x = torch.zeros(max_name_len + 1, dtype=torch.long)
                y = torch.zeros(max_name_len + 1, dtype=torch.long)
                y[:] = -1
                subname_x, subname_y = name[:i], name[:i + 1]
                if len(subname_y) > len(y):
                    subname_y = subname_y[1:]  # trim leading dot for the longest word
                x[-len(subname_x):] = torch.tensor(self.tokenizer.encode(subname_x))
                y[-len(subname_y):] = torch.tensor(self.tokenizer.encode(subname_y))
                xs.append(x)
                ys.append(y)
                # print(
                #     tokenizer.decode(x.tolist(), skip_special_tokens=False), '->',
                #     tokenizer.decode(y.tolist(), skip_special_tokens=False)
                # )
            # **** Another method, producing one example per name, e.g.:
            # **** .emma.. -> emma.__
            # **** .olivia -> olivia.
            # **** Proved to be less efficient.
            # x = torch.zeros(max_name_len + 1, dtype=torch.long)
            # y = torch.zeros(max_name_len + 1, dtype=torch.long)
            # 
            # ids = [tokenizer.token_to_id(ch) for ch in name]
            # 
            # x[0] = tokenizer.token_to_id(".")
            # x[1: 1 + len(name)] = torch.tensor(ids)
            # 
            # y[0: len(name)] = torch.tensor(tokenizer.encode(name))
            # y[len(name)] = tokenizer.token_to_id(".")
            # y[len(name) + 1 :] = tokenizer.token_to_id("_")
            # 
            # xs.append(x)
            # ys.append(y)
            # # print(
            # #     tokenizer.decode(x.tolist(), skip_special_tokens=False), '->', 
            # #     tokenizer.decode(y.tolist(), skip_special_tokens=False)
            # # )
            # **** End onther method
        self.x = torch.stack(xs)
        self.y = torch.stack(ys)

    def print_examples(
        self,
        model: torch.nn.Module,
        **kwargs,
    ):
        n_words = 15
        prompt = torch.tensor([self.tokenizer.token_to_id(".")] * self.context_len)
        for _ in range(n_words):
            x_sampled = model.generate(
                prompt,
                max_len=100,
                stop_token=self.tokenizer.token_to_id("."),
                **kwargs,
            ).to("cpu")
            text = self.tokenizer.decode(
                x_sampled[0].tolist(), skip_special_tokens=False
            )
            text = text[len(prompt) :]
            print(text)


def get_dataset(name: str, base_path: Path) -> TransformerDataset:
    if name == "names":
        return NamesDataset(base_path)
    elif name == "murakami":
        return MurakamiDataset(base_path)
    else:
        raise ValueError(f"Unknown dataset {name}")

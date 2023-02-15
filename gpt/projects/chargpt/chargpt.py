import sys
from abc import ABC, abstractmethod
from pathlib import Path

import click
import tiktoken
import torch
from torch.utils.data import Dataset, random_split
import torchtext.datasets

from gpt import utils
from gpt.model import Transformer
from gpt.trainer import Trainer


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


class SentencePieceTokenizer(Tokenizer):
    pass


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding_name="gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.enc.encode_ordinary(text), dtype=torch.long)

    def decode(self, data: torch.Tensor) -> str:
        return "".join("[" + self.enc.decode([v]) + "]" for v in data.tolist())

    def vocab_size(self) -> int:
        return self.enc.n_vocab


class TransformerDataset(Dataset):
    def __init__(self, data: torch.Tensor | str, block_size: int):
        self.block_size = block_size
        self.data = data

        # Partition the input data into a training and the test set
        # 10% of the training set, or up to 1000 examples
        n_test = min(1000, int(len(self) * 0.1))
        self.train, self.test = random_split(self, [len(self) - n_test, n_test])
        print(
            f"Created dataset of {len(self.data)} tokens, giving {len(self)} examples "
            f"with block size of {self.block_size}. Split the dataset into "
            f"{len(self.train)} training and {len(self.test)} test examples"
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y

    def __len__(self):
        return len(self.data) - self.block_size - 1


def sample_and_print(
    tokenizer: Tokenizer,
    model: Transformer,
    device: torch.device,
    top_k: int | None = None,
    max_new_tokens: int = 500,
    prompt: str = "O God, O God!",
):
    """
    Sample from the model and print the decoded text.
    """
    x_init = tokenizer.encode(prompt).to(device).view(1, -1)

    x_sampled = model.generate(
        x_init, max_new_tokens=max_new_tokens, top_k=top_k, do_sample=True
    ).to("cpu")
    # x_sampled = x_sampled[:, len(prompt) :]  # remove the prompt
    
    print('Generated text example:')
    print(tokenizer.decode(x_sampled[0]))


def create_dataset(
    tokenizer: str, input_path: Path, save_path: Path, block_size: int
) -> tuple[Tokenizer, TransformerDataset]:
    """
    Create or read the dataset from file if exists.
    """
    if tokenizer == "tiktoken":
        tokenizer = TiktokenTokenizer()
    elif tokenizer == "char":
        with input_path.open("r") as f:
            text: str = f.read()
        tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")

    if save_path.exists():
        print(f"Loading pickled dataset from {save_path}...")
        dataset = torch.load(save_path)
    else:
        print(f"Creating dataset from file {input_path}...")
        with input_path.open("r") as f:
            text: str = f.read()
        dataset = TransformerDataset(tokenizer.encode(text), block_size=block_size)
        print(f"Saving dataset to {save_path}...")
        torch.save(dataset, save_path)
    return tokenizer, dataset


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--tokenizer", default="tiktoken", type=click.Choice(["tiktoken", "char"]))
@click.option("--block-size", default=32, type=int)
@click.option("--n-layers", default=4, type=int)
@click.option("--emb-dim", default=64, type=int)
@click.option("--n-heads", default=4, type=int)
@click.option("--disable-cuda", is_flag=True)
@click.option("--resume", is_flag=True)
@click.option("--sample-only", is_flag=True)
@click.option("--batch-size", default=32, type=int)
@click.option("--learning-rate", default=5e-4, type=float)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--num-workers", default=1, type=int)
@click.option("--max-steps", default=100_000, type=int)
@click.option("--seed", default=0, type=int)
def main(
    input_file: str,
    tokenizer: str,
    block_size: int,
    n_layers: int,
    emb_dim: int,
    n_heads: int,
    disable_cuda: bool,
    resume: bool,
    sample_only: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    max_steps: int,
    seed: int,
):
    utils.set_seed(seed)
    device = utils.device(disable_cuda)

    input_path = Path(input_file)
    print(f"Building the dataset from {input_path}...")

    saves_path = Path(__file__).parent / "saves"
    saves_path.mkdir(exist_ok=True)
    
    dataset_name = f"{input_path.stem}-{tokenizer}"
    model_save_path = saves_path / f"{dataset_name}-model.pt"
    dataset_save_path = saves_path / f"{dataset_name}-dataset.pt"
    
    tokenizer, dataset = create_dataset(tokenizer, input_path, dataset_save_path, block_size=block_size)
    print()

    print("Initialising the model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size(),
        block_size=dataset.block_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    if resume or sample_only:
        if not model_save_path.exists():
            print(f"Model state not found at {model_save_path}")
            sys.exit(1)
        # Note: if we sample-only then we also assume we are resuming
        print(f"Initializing from the existing model state {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    print()
    if sample_only:
        sample_and_print(tokenizer, model=model, device=device)
        sys.exit()
        
    def _callback(*_):
        sample_and_print(tokenizer, model=model, device=device)

    trainer = Trainer(
        model=model,
        device=device,
        train_set=dataset.train,
        test_set=dataset.test,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        save_path=model_save_path,
        on_batch_end=_callback,
        on_start=_callback,
    )
    trainer.run()


if __name__ == "__main__":
    main()

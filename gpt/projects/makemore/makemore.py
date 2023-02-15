import sys
from collections import namedtuple
from pathlib import Path
import click

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split

from gpt import utils
from gpt.model import Transformer
from gpt.trainer import Trainer


class WordListDataset(TensorDataset):
    """
    List of words, tokenized on the character level
    """
    def __init__(self, words: list[str]):
        words = [w.strip() for w in words]  # Get rid of any leading/trailing space
        words = [w for w in words if w]  # Get rid of any empty strings
        
        # Our tokens are characters
        chars = sorted(list(set("".join(words))))
        self.vocab_size = len(chars) + 1  # characters followed by special 0 token
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.itos[0] = "."
        
        # Our block size would be large enough to contain the longest word
        max_word_length: int = max(len(w) for w in words)
        self.block_size = 1 + max_word_length  # <START> token followed by characters

        print(f"Number of examples in the dataset: {len(words)}")
        print(f"Max word length: {max_word_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:", "".join(chars))

        x = torch.zeros((len(words), self.block_size), dtype=torch.long)
        y = torch.zeros((len(words), self.block_size), dtype=torch.long)
        for i, word in enumerate(words):
            t = self.encode(word)
            x[i, 1 : 1 + len(t)] = t
            y[i, :-1] = x[i, 1:]
            # Index -1 will mask the loss at the inactive locations. We don't train
            # the model with the index -1: remember the embedding matrix takes x as
            # indices,
            # so index -1 wouldn't even make sense; so our vocabulary has only the "0"
            # <START> character along with real characters encoded as 1-27. Our "y" will
            # only be used in the cross_entropy function, where we explicitly pass
            # ignore_index=-1, so it doesn't look at these values in the "y" tensor.
            # When generating words, we use the prompt that wouldn't contain -1
            # characters.
            y[i, len(t) + 1 :] = -1
        super().__init__(x, y)

        self.train_set, self.test_set = self.partition()
    
    def partition(self) -> namedtuple("Splits", "train test"):
        """
        Partition the input data into a training and the test set
        10% of the training set, or up to 1000 examples
        """
        test_size: int = min(1000, int(len(self) * 0.1))
        train_set, test_set = random_split(self, [len(self) - test_size, test_size])
        print(
            f"Split the dataset into {len(train_set)} training examples "
            f"and {len(test_set)} test examples"
        )
        return train_set, test_set

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[t] for t in text], dtype=torch.long)

    def decode(self, ints: list[int]) -> str:
        return "".join(self.itos[i] for i in ints)


def create_dataset(input_path: Path, saves_dir: Path) -> WordListDataset:
    pickled_path = saves_dir / f'{input_path.stem}-dataset.pt'
    if pickled_path.exists():
        print(f"Loading pickled dataset from {pickled_path}...")
        dataset = torch.load(pickled_path)
        return dataset

    print(f"Creating dataset from file {input_path}...")
    with input_path.open("r") as f:
        text: str = f.read()
    dataset = WordListDataset(text.splitlines())

    print(f"Saving dataset to {pickled_path}...")
    torch.save(dataset, pickled_path)
    return dataset


def sample_and_print(
    dataset: WordListDataset,
    model: nn.Module,
    device: torch.device,
    top_k: int | None = None,
    clean: bool = True,
    num: int = 10,
):
    """
    Samples from the model and pretty prints the decoded samples
    """
    x_init = torch.zeros(num, 1, dtype=torch.long).to(device)

    x_sampled = model.generate(
        x_init, 
        # -1 because we already start with <START> token (index 0):
        max_new_tokens=dataset.block_size - 1, 
        top_k=top_k, 
        do_sample=True
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

        word_sample = dataset.decode(row)

        # separately track samples that we have and have not seen before
        if word_sample in dataset.train_set:
            train_samples.append(word_sample)
        elif word_sample in dataset.test_set:
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


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
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

    print(f"Building the dataset from {input_file}...")
    saves_path = Path(__file__).parent / "saves"
    saves_path.mkdir(exist_ok=True)
    input_path = Path(input_file)
    dataset = create_dataset(input_path, saves_path)
    print(f"Dataset determined that: {dataset.vocab_size=}, {dataset.block_size=}")
    print()

    print("Initialising the model...")
    model = Transformer(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    model_path = saves_path / f'{input_path.stem}-model.pt'
    if resume or sample_only:
        if not model_path.exists():
            print(f"Model state not found at {model_path}")
            sys.exit(1)
        # Note: if we sample-only then we also assume we are resuming
        print(f"Initializing from the existing model state {model_path}")
        model.load_state_dict(torch.load(model_path))
    print()
    if sample_only:
        sample_and_print(dataset, model=model, device=device)
        sys.exit()
        
    def callback(step, loss):
        sample_and_print(dataset, model=model, device=device)

    trainer = Trainer(
        model=model,
        device=device,
        train_set=dataset.train_set,
        test_set=dataset.test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        save_path=model_path,
        on_batch_end=callback,
    )
    trainer.run()


if __name__ == "__main__":
    main()

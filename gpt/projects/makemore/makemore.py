import sys
from collections import namedtuple
from pathlib import Path
import click

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

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


def create_dataset(
    input_path: Path,
    save_path: Path,
) -> WordListDataset:
    if save_path.exists():
        print(f"Loading pickled dataset from {save_path}...")
        dataset = torch.load(save_path)
    else:
        print(f"Creating dataset from file {input_path}...")
        dataset = WordListDataset(input_path.open().read().splitlines())
    
        print(
            f"Created dataset with vocab size {dataset.vocab_size}, "
            f"block size {dataset.block_size}"
        )
        print(f"Saving dataset to {save_path}...")
        torch.save(dataset, save_path)
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



class Config:
    sample_only: str = True
    number_of_samples: int = 20
    seed = 0

    # Dataset
    input_file: str = "data/names.txt"

    # Network
    emb_dim: int = 64
    n_blocks: int = 4
    n_heads: int = 4

    # Optimization
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_workers: int = 1
    max_steps: int = 100_000
    disable_cuda: bool = False
    

def main():
    utils.set_seed(Config.seed)
    device = utils.device(Config.disable_cuda)

    saves_path = Path(__file__).parent / "saves"
    saves_path.mkdir(exist_ok=True)

    input_path = Path(Config.input_file)
    dataset_name = f"{input_path.stem}"
    model_save_path = saves_path / f"{dataset_name}-model.pt"
    dataset_save_path = saves_path / f"{dataset_name}-dataset.pt"

    dataset = create_dataset(input_path, dataset_save_path)
    print()

    print("Initialising the model...")
    model = Transformer(
        vocab_size=dataset.vocab_size,
        context_len=dataset.block_size,
        emb_dim=Config.emb_dim,
        n_heads=Config.n_heads,
        n_layers=Config.n_blocks,
    )

    if model_save_path.exists():
        print(f"Initializing from the existing model state {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))

    if Config.sample_only:
        if not model_save_path.exists():
            print(f"No model file found at {model_save_path}, cannot sample")
            sys.exit(1)
        sample_and_print(dataset, model=model, device=device, num=Config.number_of_samples)
        sys.exit(0)
    print()  

    def _callback(*_):
        sample_and_print(dataset, model=model, device=device)

    trainer = Trainer(
        model=model,
        device=device,
        train_set=dataset.train_set,
        test_set=dataset.test_set,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        max_steps=Config.max_steps,
        save_path=model_save_path,
        summary_writer=SummaryWriter(),
        on_start=_callback,
        on_batch_end=_callback,
    )
    trainer.run()


if __name__ == "__main__":
    main()

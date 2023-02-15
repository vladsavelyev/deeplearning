import sys
import time
from pathlib import Path
import click

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gpt.dataset import create_dataset, DATASET_CLASSES
from gpt.model import Transformer
from gpt.trainer import Trainer


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--dataset-type", default="text",
    type=click.Choice(['text', 'word_list'], case_sensitive=False),
)
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
@click.option("--block-size", default=32, type=int)
def main(
    input_file: str,
    dataset_type: str,
    n_layers: int,
    emb_dim: int,
    n_heads: int,
    disable_cuda: str,
    resume: bool,
    sample_only: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    max_steps: int,
    seed: int,
    block_size: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Building the dataset from {input_file}...")
    input_path = Path(input_file)
    dataset = create_dataset(input_path, dataset_type, block_size=block_size)

    print(f"Dataset determined that: {dataset.vocab_size=}, {dataset.block_size=}")
    print()

    print("Initialising the model...")
    model = Transformer(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    model_path = Path(f"saves/{input_path.name}.pt")
    model_path.parent.mkdir(exist_ok=True)
    if resume or sample_only:
        # note: if we sample-only then we also assume we are resuming
        print(f"Initializing from the existing model state {model_path}")
        model.load_state_dict(torch.load(model_path))
    print()

    if sample_only:
        dataset.sample_and_print(model=model, device=device)
        sys.exit()

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
    )
    trainer.run()



if __name__ == "__main__":
    main()

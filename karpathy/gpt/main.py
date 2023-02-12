import sys
import time
from pathlib import Path
import click

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .char_dataset import CharDataset, print_samples
from .model import Transformer


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int = 50,
    max_batches: int | None = None,
):
    model.eval()
    losses = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    for batch_i, batch in enumerate(loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss)
        if max_batches is not None and batch_i >= max_batches:
            break

    out = torch.tensor(losses).mean().item()
    model.train()
    return out


@click.command()
@click.option("--input-file", type=click.Path(exists=True))
@click.option("--n-layers", default=4, type=int)
@click.option("--emb-dim", default=64, type=int)
@click.option("--n-heads", default=4, type=int)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--resume", default=False, type=bool)
@click.option("--sample-only", default=False, type=bool)
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
    device: str,
    resume: bool,
    sample_only: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    max_steps: int,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Init dataset
    input_text_path = Path(input_file)
    with input_text_path.open("r") as f:
        words: list[str] = f.read().splitlines()
    dataset = CharDataset(words)
    print(f"Dataset determined that: {dataset.vocab_size=}, {dataset.block_size=}")

    # Init model
    model = Transformer(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    model_path = "model.pt"
    if resume or sample_only:
        # note: if we sample-only then we also assume we are resuming
        print(f"Resuming from the existing model {model_path}")
        model.load_state_dict(torch.load(model_path))

    if sample_only:
        print_samples(model=model, dataset=dataset, num=50, device=device)
        sys.exit()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    data_loader = DataLoader(
        dataset.train_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=torch.utils.data.RandomSampler(
            dataset.train_set, replacement=True, num_samples=int(1e10)
        ),
    )

    writer = SummaryWriter()

    def _evaluate_and_save(best_loss: int | None = None) -> int:
        print_samples(model, dataset, device, num=10)
        train_loss, test_loss = [
            evaluate(model, subset, device, batch_size=100, max_batches=10)
            for subset in [dataset.train_set, dataset.test_set]
        ]
        writer.add_scalar("Loss/train", train_loss, step)
        writer.add_scalar("Loss/test", test_loss, step)
        writer.flush()
        print(f"Step {step} train loss: {train_loss} test loss: {test_loss}")
        # Save the model to disk if it has improved
        if best_loss is None or test_loss < best_loss:
            print(
                f"test loss {test_loss} is the best so far, saving model to "
                f"{model_path}"
            )
            torch.save(model.state_dict(), model_path)
            best_loss = test_loss
        return best_loss

    # Training loop
    best_loss = None
    step = 0
    try:
        for (x, y) in data_loader:
            t0 = time.time()

            x.to(device)
            y.to(device)

            logits, loss = model(x, y)

            # Calculate the gradient, update the weights
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Wait for all CUDA work on the GPU to finish then calculate iteration time
            # taken
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.time()

            # Logging
            if step % 10 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} | step time "
                    f"{(t1 - t0) * 1000:.2f}ms"
                )

            # Evaluate the model
            if step > 0 and step % 500 == 0:
                best_loss = _evaluate_and_save(best_loss)

            step += 1
            # Termination conditions
            if step >= max_steps:
                break
    except KeyboardInterrupt:
        print("Training interrupted by user")

    print("-" * 100)
    print("Final result:")
    _evaluate_and_save(best_loss)


if __name__ == "__main__":
    main()

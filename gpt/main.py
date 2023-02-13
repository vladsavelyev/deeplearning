import sys
import time
from pathlib import Path
import click

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gpt.dataset import create_dataset, DATASET_CLASSES
from gpt.model import Transformer


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--dataset-type", default="text",
    type=click.Choice([c.__name__ for c in DATASET_CLASSES]),
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
        dataset.sample_and_print(model=model, device=device)
        train_loss, test_loss = [
            model.evaluate(subset, device, batch_size=100, max_batches=10)
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
            x = x.to(device)
            y = y.to(device)
            
            t0 = time.time()

            logits, loss = model(x, y)

            # Calculate the gradient, update the weights
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Wait for all CUDA work on the GPU to finish then calculate iteration time
            # taken
            if device.type == "cuda":
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

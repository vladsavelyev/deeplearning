import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from picogpt import utils
from picogpt.dataset import (
    Tokenizer,
    TransformerDataset,
    CharTokenizer,
    TiktokenTokenizer,
    SentencePieceTokenizer,
    HuggingFaceTokenizer,
    split_dataset,
)
from picogpt.model import Transformer
from picogpt.trainer import Trainer


def sample_and_print(
    tokenizer: Tokenizer,
    model: Transformer,
    device: torch.device,
    top_k: int = None,
    max_new_tokens: int = 500,
    prompt: str = "Oh God! Oh God!",
):
    """
    Sample from the model and print the decoded text.
    """
    x_init = tokenizer.encode(prompt).to(device).view(1, -1)

    x_sampled = model.generate(
        x_init, max_new_tokens=max_new_tokens, top_k=top_k, do_sample=True
    ).to("cpu")
    # x_sampled = x_sampled[:, len(prompt) :]  # remove the prompt

    print("Generated text example:")
    print(tokenizer.decode(x_sampled[0]))


def create_dataset(
    tokenizer: str,
    input_path: Path,
    save_path: Path,
    context_len: int,
    vocab_size: int | None = None,
    input_test_path: Path = None,
) -> Tuple[Tokenizer, TransformerDataset]:
    """
    Create or read the dataset from file if exists.
    """
    print(f"Initializing tokenizer {tokenizer} based on {input_path}...")
    if tokenizer == "tiktoken":
        tokenizer = TiktokenTokenizer(encoding_name="gpt2")
    elif tokenizer == "char":
        tokenizer = CharTokenizer(input_path.open().read())
    elif tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer(input_path, vocab_size=vocab_size)
    elif tokenizer == "huggingface":
        tokenizer = HuggingFaceTokenizer(input_path, vocab_size=vocab_size)
    print(f"Inferred vocabulary size: {tokenizer.vocab_size()}")

    if save_path.exists():
        print(f"Loading pickled dataset from {save_path}...")
        dataset = torch.load(save_path)
    else:
        print(f"Creating dataset from file {input_path}...")
        data = tokenizer.encode(input_path.open().read())
        dataset = TransformerDataset(data, context_len)
        if input_test_path is not None:
            dataset.train = dataset
            test_data = tokenizer.encode(open(input_test_path).read())
            dataset.test = TransformerDataset(test_data, context_len)
        else:
            dataset.train, dataset.test = split_dataset(dataset)
        print(
            f"Created dataset of {len(dataset.data)} tokens, giving {len(dataset)} "
            f"examples with context length {dataset.context_len}. Split the dataset "
            f"into {len(dataset.train)} training and {len(dataset.test)} test examples"
        )
        print(f"Saving dataset to {save_path}...")
        torch.save(dataset, save_path)
    return tokenizer, dataset


datasets_path = (
    next(
        Path(p)
        for p in ["/Users/vlad/googledrive", "/content/drive/MyDrive"]
        if Path(p).exists()
    )
    / "AI/datasets"
)
runs_saves_root = next(
    Path(p)
    for p in ["/content/drive/MyDrive/AI/hipogpt", Path(__file__).parent]
    if Path(p).exists()
)
saves_path = runs_saves_root / "saves"
runs_path = runs_saves_root / "runs"
saves_path.mkdir(exist_ok=True)
runs_path.mkdir(exist_ok=True)


class Config:
    sample_only: str = False
    seed = 0

    # Dataset
    tokenizer: str = "sentencepiece"
    input_path = datasets_path / "tinyshakespeare" / "tinyshakespeare.txt"
    input_test_file = None
    vocab_size: int = 2000

    # Network
    context_len: int = 32
    emb_dim: int = 32
    n_blocks: int = 2
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

    dataset_name = f"{Config.input_path.stem}-{Config.tokenizer}"
    model_save_path = saves_path / f"{dataset_name}-model.pt"
    dataset_save_path = saves_path / f"{dataset_name}-dataset.pt"

    tokenizer, dataset = create_dataset(
        tokenizer=Config.tokenizer,
        input_path=Config.input_path,
        input_test_path=Path(Config.input_test_file) if Config.input_test_file else None,
        save_path=dataset_save_path,
        context_len=Config.context_len,
        vocab_size=Config.vocab_size,
    )
    print()

    print("Initialising the model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size(),
        context_len=dataset.context_len,
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
        sample_and_print(tokenizer, model=model.to(device), device=device)
        sys.exit(0)
    print()

    def _callback(*_):
        sample_and_print(tokenizer, model=model, device=device)

    trainer = Trainer(
        model=model,
        device=device,
        train_set=dataset.train,
        test_set=dataset.test,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        max_steps=Config.max_steps,
        save_path=model_save_path,
        summary_writer=SummaryWriter(),
        on_batch_end=_callback,
        on_start=_callback,
    )
    trainer.run()


if __name__ == "__main__":
    main()

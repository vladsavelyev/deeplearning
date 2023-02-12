import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, words: list[str]):
        words = self._clean_words(words)
        self.words = words

        chars = sorted(list(set("".join(words))))
        self.vocab_size = len(chars) + 1  # characters followed by special 0 token

        max_word_length: int = max(len(w) for w in words)
        self.block_size = 1 + max_word_length  # <START> token followed by characters

        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

        print(f"Number of examples in the dataset: {len(words)}")
        print(f"Max word length: {max_word_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:", "".join(chars))

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

    def encode(self, word) -> torch.Tensor:
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ints: list[int]) -> str:
        return "".join(self.itos[i] for i in ints)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index: int):
        word: str = self.words[index]
        w: torch.Tensor = self.encode(word)

        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        x[1 : 1 + len(w)] = w
        y[: len(w)] = w
        y[len(w) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y


def print_samples(
    model,
    dataset: CharDataset,
    device: str,
    num: int = 10,
    top_k: int | None = None,
):
    """
    Samples from the model and pretty prints the decoded samples
    """
    x_init = torch.zeros(num, 1, dtype=torch.long).to(device)

    top_k = top_k if top_k != -1 else None

    # -1 because we already start with <START> token (index 0)
    n_steps = dataset.block_size - 1

    x_sampled = model.generate(
        x_init, max_new_tokens=n_steps, top_k=top_k, do_sample=True
    ).to("cpu")

    train_samples, test_samples, new_samples = [], [], []

    for sample_i in range(x_sampled.shape[0]):
        # Get the sample_i'th row of sampled integers, as a Python list.
        # We are also trimming out the first <START> token.
        row: list[int] = x_sampled[sample_i, 1:].tolist()

        # Token "0" is the <STOP> token, so we crop the output sequence at that point.
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
        print(f"{len(samples)} samples that are {desc}:", " ".join(samples))

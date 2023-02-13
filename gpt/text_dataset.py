import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        self.text = text

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

        print(f"Text size: {len(text)}")
        print(f"Block size: {self.block_size}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:", "".join(chars))

        print('Converting the text into tensors...')
        self.x = torch.zeros((len(text) - block_size, block_size), dtype=torch.long)
        self.y = torch.zeros((len(text) - block_size, block_size), dtype=torch.long)
        for char_idx in range(len(text) - block_size):
            wx = text[char_idx:char_idx + block_size]
            wy = text[char_idx + 1:char_idx + block_size + 1]
            self.x[char_idx, :] = self.encode(wx)
            self.y[char_idx, :] = self.encode(wy)
        
        print('Splitting into train and test...')
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
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def sample_and_print(
        self,
        model,
        device: str,
        top_k: int | None = None,
        clean: bool = False,
        max_new_tokens: int = 500,
        prompt: str = '\n'
    ):
        """
        Samples from the model and pretty prints the decoded samples
        """
        x_init = self.encode(prompt).to(device).view(1, -1)
    
        x_sampled = model.generate(
            x_init, max_new_tokens=max_new_tokens, top_k=top_k, do_sample=True
        ).to("cpu")
        x_sampled = x_sampled[:, len(prompt):]  # remove the prompt

        # Get the sample_i'th row of sampled integers, as a Python list
        row: list[int] = x_sampled[0].tolist()
        print(self.decode(row))

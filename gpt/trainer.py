import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        train_set: Subset, 
        test_set: Subset, 
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        max_steps: int,
        on_batch_end: Callable[[int, float], None] | None = None,
        on_start: Callable[[], None] | None = None,
        save_path: Path | None = None,
        summary_writer: SummaryWriter | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_set = train_set
        self.test_set = test_set
        self.data_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            pin_memory=True,
            sampler=torch.utils.data.RandomSampler(
                train_set, replacement=True, num_samples=int(1e10)
            ),
            num_workers=num_workers,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.writer = summary_writer
        self.max_steps = max_steps
        self.on_batch_end = on_batch_end
        self.on_start = on_start
        self.save_path = save_path

        # Will be populated during training:
        self.best_loss = None
    
    @torch.inference_mode()
    def _checkpoint(self, step: int):
        """
        Checkpoint to more accurately evaluate the model, print stats, update best_loss,
        save the model state, and optionally call user callbacks.
        """
        train_loss, test_loss = [
            self.model.evaluate(subset, self.device, batch_size=100, max_batches=10)
            for subset in [self.train_set, self.test_set]
        ]
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, step)
            self.writer.add_scalar("Loss/test", test_loss, step)
            self.writer.flush()
        print(f"Step {step} train loss: {train_loss} test loss: {test_loss}")

        # Save the model to disk if it has improved
        if self.best_loss is None or test_loss < self.best_loss:
            self.best_loss = test_loss
            if self.save_path is not None:
                print(
                    f"test loss {test_loss} is the best so far, saving model to "
                    f"{self.save_path}"
                )
                torch.save(self.model.state_dict(), self.save_path)

        if self.on_batch_end:
            self.on_batch_end(step, test_loss)

    def run(self):
        print(f"Training model on device '{self.device}'")
        self.model.train()
        step = 0
        if self.on_start:
            self.on_start()
        try:
            for (x, y) in self.data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                t0 = time.time()
    
                logits, loss = self.model(x, y)
    
                # Calculate the gradient, update the weights
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
    
                # Wait for all CUDA work on the GPU to finish then calculate iteration time
                # taken
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
    
                # Logging
                if step % 10 == 0:
                    print(
                        f"step {step} | loss {loss.item():.4f} | step time "
                        f"{(t1 - t0) * 1000:.2f}ms"
                    )
    
                if step > 0 and step % 500 == 0:
                    self._checkpoint(step)

                step += 1
                # Termination conditions
                if self.max_steps and step >= self.max_steps:
                    break

        except KeyboardInterrupt:
            print("Training interrupted by user")
    
        print("-" * 100)
        print("Final result:")
        self._checkpoint(step)

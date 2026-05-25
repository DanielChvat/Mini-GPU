#!/usr/bin/env python3
"""Train a small MNIST conv net on the Mini-GPU backend.

Example:
    PYTHONPATH=$PWD python demos/mnist_conv_train.py --port /dev/ttyUSB1 --epochs 1
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import torch
import torch_mini_gpu


class TinyMnistConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=3, stride=4, bias=True)
        self.hidden = torch.nn.Linear(4 * 7 * 7, 16, bias=True)
        self.fc = torch.nn.Linear(16, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.hidden(x))
        return self.fc(x)


def load_dataset(root: Path, download: bool):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("This demo needs torchvision installed for MNIST loading.") from exc

    try:
        return datasets.MNIST(
            root=str(root),
            train=True,
            download=download,
            transform=transforms.ToTensor(),
        )
    except RuntimeError as exc:
        if not download:
            raise SystemExit(
                f"MNIST was not found under {root}. Re-run with --download or pass --data-root."
            ) from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Mini-GPU serial port, for example /dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--memory-size", type=int, default=65536)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download MNIST if it is not already present")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps; 0 means full dataset")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--window", type=int, default=25, help="Rolling window, in optimizer steps, for logged loss/accuracy")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=Path("demos/mnist_conv_weights.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch_mini_gpu.connect(args.port, baud=args.baud, memory_size=args.memory_size)

    try:
        dataset = load_dataset(args.data_root, args.download)
        generator = torch.Generator().manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model = TinyMnistConv().to("minigpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, foreach=False)
        initial_weight = model.fc.weight.detach().to("cpu").clone()

        try:
            print(
                f"training MNIST samples={len(dataset)} batch_size={args.batch_size} "
                f"epochs={args.epochs} lr={args.lr} max_steps={args.max_steps or 'all'}"
            )
            global_step = 0
            rolling_losses: deque[tuple[float, int]] = deque(maxlen=args.window)
            rolling_correct: deque[tuple[int, int]] = deque(maxlen=args.window)
            rolling_grad_max: deque[float] = deque(maxlen=args.window)
            for epoch in range(args.epochs):
                total_loss = 0.0
                correct = 0
                seen = 0

                for image, label in loader:
                    logits = model(image.to("minigpu"))
                    loss = torch.ops.minigpu.cross_entropy(logits, label.to(torch.int32).to("minigpu"))
                    loss.backward()
                    grad_max = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_cpu = param.grad.detach().to("cpu")
                            param_grad_max = torch.max(torch.abs(grad_cpu)).item()
                            grad_max = max(grad_max, float(param_grad_max))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    preds = torch.argmax(logits.detach().to("cpu"), dim=-1)
                    batch_correct = int((preds == label).sum().item())
                    loss_value = float(loss.detach().to("cpu"))
                    correct += batch_correct
                    total_loss += loss_value * float(image.size(0))
                    seen += int(image.size(0))
                    rolling_losses.append((loss_value, int(image.size(0))))
                    rolling_correct.append((batch_correct, int(image.size(0))))
                    rolling_grad_max.append(grad_max)
                    global_step += 1

                    if args.log_every > 0 and global_step % args.log_every == 0:
                        window_seen = sum(count for _, count in rolling_losses)
                        window_loss = sum(value * count for value, count in rolling_losses) / float(window_seen)
                        window_acc = sum(value for value, _ in rolling_correct) / float(window_seen)
                        window_grad_max = max(rolling_grad_max) if rolling_grad_max else 0.0
                        print(
                            f"step={global_step:05d} epoch={epoch + 1:03d} "
                            f"loss={loss_value:.6f} "
                            f"window_loss={window_loss:.6f} "
                            f"window_acc={window_acc:.3f} "
                            f"grad_max={window_grad_max:.3e} "
                            f"acc={correct / float(seen):.3f}"
                        )
                    if args.max_steps and global_step >= args.max_steps:
                        break

                avg_loss = total_loss / float(seen)
                accuracy = correct / float(seen)
                print(
                    f"epoch={epoch + 1:03d} avg_loss={avg_loss:.6f} "
                    f"train_acc={accuracy:.3f}"
                )
                if args.max_steps and global_step >= args.max_steps:
                    break
            final_weight = model.fc.weight.detach().to("cpu")
            weight_delta = torch.max(torch.abs(final_weight - initial_weight)).item()
            print(f"max_fc_weight_delta={weight_delta:.6e}")
            if args.save_path:
                args.save_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_state = {
                    name: tensor.detach().to("cpu")
                    for name, tensor in model.state_dict().items()
                }
                torch.save(
                    {
                        "model": cpu_state,
                        "epochs": args.epochs,
                        "global_step": global_step,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                    },
                    args.save_path,
                )
                print(f"saved_weights={args.save_path}")
        except KeyboardInterrupt:
            print("\nInterrupted; closing Mini-GPU transport.")
    finally:
        try:
            torch_mini_gpu.disconnect()
        except KeyboardInterrupt:
            print("Disconnect interrupted; the board may need a reset before the next run.")


if __name__ == "__main__":
    main()

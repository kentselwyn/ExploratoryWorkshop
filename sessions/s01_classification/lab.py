import yaml
import argparse

import torch
import torch.nn as nn

from typing import Any
from pathlib import Path
from easytrain import BaseModel, BaseDataset, DatasetProvider, BaseTask, Trainer, StepOutput
from torchvision import datasets, transforms

from sessions.common import Lab, run, write_json

import matplotlib.pyplot as plt

class MNISTDataProvider(DatasetProvider):
    def __init__(
        self,
        root: str = "data",
        download: bool = True,
        validation_size: int = 5000,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = str(Path(root).expanduser())
        self.download = download
        self.validation_size = validation_size
        self.split_seed = split_seed
        self._datasets: dict[str, BaseDataset] = {}

    def get_dataset(self, split: str) -> BaseDataset:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown MNIST split: {split!r}")
        if split not in self._datasets:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            try:
                dataset = datasets.MNIST(
                    self.root, train=split != "test", download=self.download,
                    transform=transform,
                )
            except (RuntimeError, OSError) as exc:
                raise RuntimeError(
                    f"Could not load MNIST at {self.root!r}. Enable download in "
                    "the config (or use --download), check network access, or "
                    "point --data-dir to an existing MNIST download."
                ) from exc
            if split == "test":
                self._datasets["test"] = dataset
            else:
                if self.validation_size >= len(dataset):
                    raise ValueError("validation_size must be smaller than the training set")
                train, valid = torch.utils.data.random_split(
                    dataset, [len(dataset) - self.validation_size, self.validation_size],
                    generator=torch.Generator().manual_seed(self.split_seed),
                )
                self._datasets.update(train=train, val=valid)
        return self._datasets[split]


class ClassificationTask(BaseTask):
    def step(
        self,
        model: torch.nn.Module, 
        criterion: torch.nn.Module,
        batch: Any,
    ):
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return StepOutput(
            loss=loss,
            loss_dict={"loss": loss.item()},
            weight=targets.numel(),
            pred=outputs,
            trgt=targets
        )

    
    def reset_metrics(self) -> None:
        self.absolute_error = 0.0
        self.count = 0

    
    def update_metrics(self, output: StepOutput) -> None:
        self.absolute_error += (output.pred.detach().argmax(1) - output.trgt).abs().sum().item()
        self.count += output.trgt.numel()


    def compute_metrics(self) -> dict[str, float]:
        return {"mae": self.absolute_error / self.count}

    
class MNISTCNN(BaseModel):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build(config: dict[str, Any]) -> Lab:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provider = MNISTDataProvider(
        root="data",
        download=True,
        validation_size=5000,
        split_seed=42,
    )
    datasets = {
        split: provider.get_dataset(split) for split in ["train", "val", "test"]
    }
    model = MNISTCNN(num_classes=10).to(device)
    return Lab(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        datasets=datasets,
        task=ClassificationTask(),
    )


@torch.inference_mode()
def artifacts(lab, test_loader, output_dir, device):
    lab.model.eval()
    examples_per_digit = 4
    examples = {
        outcome: {digit: [] for digit in range(10)}
        for outcome in ("correct", "incorrect")
    }
    preview = []
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for images, labels in test_loader:
        predicted = lab.model(images.to(device)).argmax(1).cpu()
        labels = labels.cpu()
        confusion += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
        for image, label, prediction in zip(images, labels.tolist(), predicted.tolist()):
            outcome = "correct" if prediction == label else "incorrect"
            group = examples[outcome][label]
            if len(preview) >= 32 and len(group) >= examples_per_digit:
                continue
            sample = (image.detach().cpu().clone(), prediction, label)
            if len(preview) < 32:
                preview.append(sample)
            if len(group) < examples_per_digit:
                group.append(sample)
    write_json(output_dir / "confusion.json", confusion.tolist())

    def plot_predictions(samples, path, *, columns, title):
        rows = max(1, (len(samples) + columns - 1) // columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(columns * 1.8, rows * 2), squeeze=False,
        )
        for axis in axes.flat:
            axis.axis("off")
            axis.set_box_aspect(1)
        for axis, (image, prediction, label) in zip(axes.flat, samples):
            if image is None:
                axis.text(0.5, 0.5, "No example available", ha="center", va="center", fontsize=9)
                axis.set_title(f"True: {label}", fontsize=9)
            else:
                axis.imshow(
                    (image.squeeze(0) * 0.3081 + 0.1307).clamp(0, 1),
                    cmap="gray", vmin=0, vmax=1,
                )
                axis.set_title(f"Pred: {prediction} | True: {label}", fontsize=9)
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(path, dpi=150)
        plt.close(figure)

    plot_predictions(preview, output_dir / "predictions.png", columns=8, title="Test predictions")
    write_json(output_dir / "predictions.json", {
        "labels": [label for _, _, label in preview],
        "predictions": [prediction for _, prediction, _ in preview],
    })

    for outcome, by_digit in examples.items():
        samples = []
        for digit, group in by_digit.items():
            samples.extend(group)
            samples.extend([(None, None, digit)] * (examples_per_digit - len(group)))
        plot_predictions(
            samples, output_dir / f"{outcome}_predictions.png",
            columns=examples_per_digit, title=f"{outcome.capitalize()} predictions by true digit",
        )


if __name__ == "__main__":
    run("s01_classification")

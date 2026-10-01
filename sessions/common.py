from dataclasses import dataclass

import json
import time
import yaml
import importlib

import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from pathlib import Path
from easytrain import Trainer, Callback
from easytrain.runtime import write_json

from torchvision.utils import save_image

@dataclass
class Lab:
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    task: object
    datasets: dict
    scheduler: object = None
    scheduler_interval: str = "epoch"
    collate_fn: object = None
    callbacks: tuple = ()


class Progress(Callback):
    def on_epoch_end(self, trainer, metrics):
        print(json.dumps(metrics), flush=True)


def run(session, *, config=None):
    if config is None:
        directory = Path(__file__).parent / session
        config_path = directory / "base.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    module = importlib.import_module(f"sessions.{session}.lab")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    lab = module.build(config)
    trainer = Trainer(
        model=lab.model,
        criterion=lab.criterion,
        optimizer=lab.optimizer,
        task=lab.task,
        device=device,
        output_dir=Path(config["output_dir"]),
        callbacks=[*lab.callbacks, Progress()],
    )
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset=lab.datasets[split],
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            num_workers=config.get("num_workers", 4),
        )
        for split in ["train", "val", "test"]
    }

    fit_beg = time.perf_counter()
    trainer.fit(
        train_loader=loaders["train"],
        valid_loader=loaders["val"],
        epochs=config["epochs"],
    )
    fit_end = time.perf_counter()
    plot_history(trainer.history, Path(config["output_dir"]) / "loss_history.png")

    trainer.load_checkpoint(path=Path(config["output_dir"]) / "checkpoints" / "best.pt", weights_only=True)

    eval_beg = time.perf_counter()
    metrics = trainer.evaluate(test_loader=loaders["test"])
    eval_end = time.perf_counter()

    module.artifacts(lab, loaders["test"], Path(config["output_dir"]), device)
    result = {
        "session": session,
        "fit_time": round(fit_end - fit_beg, 2),
        "eval_time": round(eval_end - eval_beg, 2),
        "metrics": metrics,
        "best_epoch": trainer.best_epoch,
    }
    write_json(Path(config["output_dir"]) / "results.json", result)
    print(yaml.safe_dump(result, sort_keys=False))


# Utility function
def save_grid(images, path, *, columns=8):
    save_image(images.detach().float().cpu().clamp(0, 1), path, nrow=columns)


# Plot history
def plot_history(history, path):
    figure, axes = plt.subplots(figsize=(6, 4))
    for split in ("train", "val"):
        axes.plot([row["epoch"] for row in history], [row[f"{split}_metrics"]["loss"] for row in history], label=split)
    axes.set(xlabel="Epoch", ylabel="Loss")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

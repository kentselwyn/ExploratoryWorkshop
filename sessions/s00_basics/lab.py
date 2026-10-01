"""Run clean and noisy line-fitting experiments: python -m sessions.s00_basics.lab."""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from easytrain import BaseDataset, BaseModel, BaseTask, Callback, DatasetProvider, StepOutput
from PIL import Image

from sessions.common import Lab, run, write_json

import matplotlib.pyplot as plt


class LineDataset(BaseDataset):
    def __init__(
        self, size, *, true_m, true_b, x_min, x_max, noise_std, seed,
    ):
        if size <= 0 or x_min >= x_max or noise_std < 0:
            raise ValueError("Require positive dataset size, x_min < x_max, and noise_std >= 0")
        self.true_m, self.true_b = true_m, true_b
        self.x_min, self.x_max = x_min, x_max
        self.noise_std = noise_std
        self.x = x_min + (x_max - x_min) * torch.rand(
            size, 1, generator=torch.Generator().manual_seed(seed),
        )
        noise = torch.randn(size, 1, generator=torch.Generator().manual_seed(seed + 1000))
        self.y = true_m * self.x + true_b + noise_std * noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class LineDataProvider(DatasetProvider):
    def __init__(
        self, *, train_size, val_size, test_size, seed,
        true_m, true_b, x_min, x_max, noise_std,
    ):
        self._datasets = {
            split: LineDataset(
                size, true_m=true_m, true_b=true_b, x_min=x_min, x_max=x_max,
                noise_std=noise_std, seed=seed + offset,
            )
            for offset, (split, size) in enumerate(
                (("train", train_size), ("val", val_size), ("test", test_size))
            )
        }

    def get_dataset(self, split):
        if split not in self._datasets:
            raise ValueError(f"Unknown line-data split: {split!r}")
        return self._datasets[split]


class LineModel(BaseModel):
    def __init__(self, m=0.0, b=0.0):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(float(m)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def forward(self, x):
        return self.m * x + self.b


class RegressionTask(BaseTask):
    def step(self, model, criterion, batch):
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return StepOutput(
            loss=loss, loss_dict={"loss": loss.item()}, weight=targets.numel(),
            pred=outputs, trgt=targets,
        )

    def reset_metrics(self):
        self.squared_error = 0.0
        self.count = 0

    def update_metrics(self, output):
        self.squared_error += (output.pred.detach() - output.trgt).square().sum().item()
        self.count += output.trgt.numel()

    def compute_metrics(self):
        return {"mse": self.squared_error / self.count}


@torch.inference_mode()
def plot_fit(axis, dataset, model, device, x, y):
    line_x = torch.linspace(dataset.x_min, dataset.x_max, 200).reshape(-1, 1)
    true_y = dataset.true_m * line_x + dataset.true_b
    fitted_y = model(line_x.to(device)).cpu()
    axis.scatter(x.flatten(), y.flatten(), s=16, alpha=0.6, label="Data")
    axis.plot(line_x.flatten(), true_y.flatten(), "--", color="darkorange", label="Ground truth")
    fitted_line, = axis.plot(
        line_x.flatten(), fitted_y.flatten(), color="crimson", label="Model fit",
    )
    values = torch.cat((y.flatten(), true_y.flatten(), fitted_y.flatten()))
    padding = max((values.max() - values.min()).item() * 0.1, 0.1)
    axis.set(
        xlabel="x", ylabel="y", xlim=(dataset.x_min, dataset.x_max),
        ylim=(values.min().item() - padding, values.max().item() + padding),
        title=f"Ground truth: y = {dataset.true_m:g}x {dataset.true_b:+g}; noise σ = {dataset.noise_std:g}",
    )
    axis.legend(loc="upper left")
    return line_x, fitted_line


class StepPlot(Callback):
    def __init__(self, dataset, *, total_steps, fps=10):
        if fps <= 0:
            raise ValueError("gif_fps must be positive")
        self.dataset = dataset
        self.total_steps = total_steps
        self.fps = fps

    def on_fit_start(self, trainer):
        self.output_dir = Path(trainer.output_dir)
        self.steps_dir = self.output_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        for path in self.steps_dir.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9].png"):
            path.unlink()
        self.history = []
        self.frame_paths = []
        self.figure, (fit_axis, self.loss_axis) = plt.subplots(
            1, 2, figsize=(10, 4.5), constrained_layout=True,
        )
        self.line_x, self.fitted_line = plot_fit(
            fit_axis, self.dataset, trainer.model, trainer.device,
            self.dataset.x, self.dataset.y,
        )
        self.loss_line, = self.loss_axis.plot([], [], color="crimson")
        self.loss_axis.set(
            xlabel="Optimizer step", ylabel="MSE (log scale)", yscale="log",
            xlim=(0, max(1, self.total_steps)), title="Full training-set MSE",
        )
        self.loss_axis.grid(alpha=0.2)
        self.record_step(trainer)

    @torch.inference_mode()
    def record_step(self, trainer):
        predictions = trainer.model(self.dataset.x.to(trainer.device))
        mse = nn.functional.mse_loss(predictions, self.dataset.y.to(trainer.device)).item()
        m, b = trainer.model.m.item(), trainer.model.b.item()
        self.history.append({"step": trainer.global_step, "m": m, "b": b, "mse": mse})
        self.fitted_line.set_ydata(trainer.model(self.line_x.to(trainer.device)).cpu().flatten())
        self.loss_line.set_data(
            [row["step"] for row in self.history],
            [max(row["mse"], 1e-12) for row in self.history],
        )
        if len(self.history) == 1:
            self.loss_axis.set_ylim(1e-12, max(mse * 1.25, 1e-11))
        case = "Clean" if self.dataset.noise_std == 0 else "Noisy"
        self.figure.suptitle(
            f"{case} data | Step {trainer.global_step}\n"
            f"m = {m:.4f}, b = {b:.4f}, MSE = {mse:.4g}",
        )
        path = self.steps_dir / f"step_{trainer.global_step:06d}.png"
        self.figure.savefig(path, dpi=100)
        self.frame_paths.append(path)

    def on_after_optimizer_step(self, trainer):
        self.record_step(trainer)

    def on_fit_end(self, trainer):
        frames = []
        try:
            write_json(self.output_dir / "step_history.json", self.history)
            for path in self.frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
            frames[0].save(
                self.output_dir / "training.gif", save_all=True, append_images=frames[1:],
                duration=round(1000 / self.fps), loop=0, disposal=2, optimize=False,
            )
        finally:
            for frame in frames:
                frame.close()
            plt.close(self.figure)


def build(config: dict[str, Any]) -> Lab:
    torch.manual_seed(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    provider = LineDataProvider(**{
        name: config[name] for name in (
            "train_size", "val_size", "test_size", "seed", "true_m", "true_b",
            "x_min", "x_max", "noise_std",
        )
    })
    datasets = {split: provider.get_dataset(split) for split in ("train", "val", "test")}
    model = LineModel(config["initial_m"], config["initial_b"]).to(device)
    return Lab(
        model=model, criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=config["learning_rate"]),
        datasets=datasets, task=RegressionTask(),
        callbacks=(StepPlot(
            datasets["train"], fps=config["gif_fps"],
            total_steps=math.ceil(len(datasets["train"]) / config["batch_size"]) * config["epochs"],
        ),),
    )


@torch.inference_mode()
def artifacts(lab, test_loader, output_dir, device):
    lab.model.eval()
    batches = list(test_loader)
    x = torch.cat([inputs for inputs, _ in batches]).cpu()
    y = torch.cat([targets for _, targets in batches]).cpu()
    predictions = lab.model(x.to(device)).cpu()
    dataset = lab.datasets["test"]
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_fit(axis, dataset, lab.model, device, x, y)
    mse = nn.functional.mse_loss(predictions, y).item()
    figure.suptitle(
        f"Best validation checkpoint on test data\n"
        f"m = {lab.model.m.item():.4f}, b = {lab.model.b.item():.4f}, MSE = {mse:.4g}",
    )
    figure.savefig(output_dir / "final_fit.png", dpi=150)
    plt.close(figure)
    write_json(output_dir / "parameters.json", {
        "ground_truth": {"m": dataset.true_m, "b": dataset.true_b},
        "learned": {"m": lab.model.m.item(), "b": lab.model.b.item()},
        "noise_std": dataset.noise_std,
        "test_mse": mse,
    })


def main():
    with (Path(__file__).parent / "base.yaml").open() as file:
        config = yaml.safe_load(file)
    for case, noise_std in (("clean", 0.0), ("noisy", config["noise_std"])):
        print(f"Training the {case} case", flush=True)
        run("s00_basics", config={
            **config, "noise_std": noise_std,
            "output_dir": str(Path(config["output_dir"]) / case),
        })


if __name__ == "__main__":
    main()

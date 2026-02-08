import subprocess
import sys

import click
import torch

from tuna.inference.pipeline import InferencePipeline
from tuna.inference.predictor import Predictor


@click.group()
def main():
    """TUnA inference CLI."""
    pass


@main.command()
@click.argument("seq_a")
@click.argument("seq_b")
@click.option(
    "--model",
    type=str,
    default="tuna",
    show_default=True,
    help="Model to use (HF repo id, local path, or one of: tuna, tfc, esm_mlp, esm_gp).",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Device to run on.",
)
def predict(model, seq_a, seq_b, device):
    """Quickly make a prediction for a single sequence pair."""
    predictor = Predictor.from_pretrained(repo_or_dir=model, device=device)
    pipeline = InferencePipeline(predictor)
    score, uncertainty = pipeline.predict_pair(seq_a, seq_b)
    click.echo(f"Score: {score:.3f}, Uncertainty: {uncertainty:.3f}")


@main.command()
@click.option(
    "--model",
    type=str,
    default="tuna",
    show_default=True,
    help="Model config name (Hydra).",
)
@click.option(
    "--dataset",
    type=str,
    default="gold-standard",
    show_default=True,
    help="Dataset config name (Hydra).",
)
@click.option(
    "--max-epochs",
    type=int,
    default=None,
    help="Override trainer.max_epochs.",
)
@click.option(
    "--config-name",
    type=str,
    default=None,
    help="Hydra config name override (optional).",
)
@click.option(
    "--config-path",
    type=str,
    default=None,
    help="Hydra config path override (optional).",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help="Additional Hydra overrides (repeatable).",
)
def train(model, dataset, max_epochs, config_name, config_path, overrides):
    """Train a model using Hydra/Lightning (wrapper around train.py)."""
    cmd = [sys.executable, "train.py", f"model={model}", f"dataset={dataset}"]
    if max_epochs is not None:
        cmd.append(f"trainer.max_epochs={max_epochs}")
    if config_name is not None:
        cmd.extend(["--config-name", config_name])
    if config_path is not None:
        cmd.extend(["--config-path", config_path])
    if overrides:
        cmd.extend(list(overrides))

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

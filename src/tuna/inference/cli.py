import click
import torch

from tuna.inference.wrappers import InferenceWrapper


@click.group()
def main():
    """TUnA inference CLI."""
    pass


@main.command()
@click.option(
    "--model",
    type=str,
    default="forge-e38-fid=0.47",
    show_default=True,
    help="Model to use.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for inference. Warning: larger batch sizes require more memory.",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Device to run on.",
)
def predict(model, device):
    """Predict the interaction between two proteins."""
    wrapper = InferenceWrapper.from_pretrained(
        repo_or_dir=model,
        device=device,
        map_location=device,
    )
    click.echo(wrapper.predict())


if __name__ == "__main__":
    main()

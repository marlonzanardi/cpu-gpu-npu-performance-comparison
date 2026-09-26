from __future__ import annotations

import click

from benchmark.hardware import collect_machine, repo_root
from benchmark.runner import execute

MODELS = ("mobilenet_v2", "resnet50")
PRECISIONS = ("float16", "float32")


@click.group()
def cli() -> None:
    """Compare one Core ML model on CPU, GPU, and Neural Engine."""


@cli.command("run")
@click.option("--model", "models", multiple=True, type=click.Choice(MODELS))
@click.option("--precision", "precisions", multiple=True, type=click.Choice(PRECISIONS))
@click.option("--batch", "batches", multiple=True, type=int)
@click.option("--warmup", default=20, show_default=True, type=int)
@click.option("--iterations", default=100, show_default=True, type=int)
@click.option("--sessions", default=3, show_default=True, type=int)
@click.option("--pause-seconds", default=2.0, show_default=True, type=float)
def run(
    models: tuple[str, ...],
    precisions: tuple[str, ...],
    batches: tuple[int, ...],
    warmup: int,
    iterations: int,
    sessions: int,
    pause_seconds: float,
) -> None:
    """Convert the models, record where Core ML places them, and measure latency."""
    machine = collect_machine()
    chip = machine["chip"]
    power = machine["power_source"]
    click.echo(f"{chip}  power={power}  low_power_mode={machine['low_power_mode']}")
    if power == "battery":
        click.echo("Running on battery. Frequency and thermals can differ from AC power.", err=True)
    if machine["low_power_mode"] is True:
        click.echo("Low Power Mode is on. Results are not comparable to a full-power run.", err=True)
    try:
        execute(
            repo_root(),
            machine,
            models or MODELS,
            precisions or PRECISIONS,
            batches or (1,),
            warmup,
            iterations,
            sessions,
            pause_seconds,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

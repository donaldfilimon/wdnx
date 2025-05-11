import json

import click
from flask import Blueprint
from flask.cli import with_appcontext

from lylex.training import TrainingManager

# CLI plugin blueprint for machine learning tasks
bp = Blueprint("ml_tasks", __name__, cli_group="ml")


@bp.cli.command("train")
@click.argument("config_file", type=click.Path(exists=True))
@with_appcontext
def train_cli(config_file):
    """
    Train a model based on a JSON configuration file.
    """
    cfg = json.loads(open(config_file, "r", encoding="utf-8").read())
    tm = TrainingManager(
        model_name=cfg["model_name"],
        backend=cfg.get("backend", "pt"),
        peft_r=cfg.get("peft_r", 8),
        peft_alpha=cfg.get("peft_alpha", 16),
        peft_dropout=cfg.get("peft_dropout", 0.1),
        wandb_project=cfg.get("wandb_project"),
    )
    tm.train(
        train_dataset=cfg["train_dataset"],
        eval_dataset=cfg.get("eval_dataset"),
        output_dir=cfg.get("output_dir", "./trained_model"),
        **cfg.get("training_args", {}),
    )
    click.echo(f"Training complete. Model saved to {cfg.get('output_dir', './trained_model')}")


@bp.cli.command("tune")
@click.argument("config_file", type=click.Path(exists=True))
@with_appcontext
def tune_cli(config_file):
    """
    Perform hyperparameter search based on a JSON configuration file.
    """
    cfg = json.loads(open(config_file, "r", encoding="utf-8").read())
    tm = TrainingManager(
        model_name=cfg["model_name"],
        backend=cfg.get("backend", "pt"),
        wandb_project=cfg.get("wandb_project"),
    )
    best = tm.hyperparameter_search(
        train_dataset=cfg["train_dataset"],
        eval_dataset=cfg.get("eval_dataset"),
        output_dir=cfg.get("output_dir", "./hyperopt_model"),
        n_trials=cfg.get("n_trials", 10),
        direction=cfg.get("direction", "minimize"),
        **cfg.get("training_args", {}),
    )
    click.echo(json.dumps(best, indent=2))

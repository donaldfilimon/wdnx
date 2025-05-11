import logging
import os
from typing import Any, Dict, Optional

import optuna
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback

from .ai import LylexModelHandler

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    High-level manager for advanced training workflows with hyperparameter tuning, LoRA, and Weights & Biases logging.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "pt",
        peft_r: int = 8,
        peft_alpha: int = 16,
        peft_dropout: float = 0.1,
        wandb_project: Optional[str] = None,
    ):
        self.handler = LylexModelHandler(backend)
        self.handler.load_model(model_name)
        # Apply LoRA to the model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=peft_r,
            lora_alpha=peft_alpha,
            lora_dropout=peft_dropout,
        )
        self.handler.model = get_peft_model(self.handler.model, peft_config)
        # Initialize W&B if project provided
        if wandb_project:
            wandb.init(project=wandb_project)
            self.wandb_project = wandb_project
        else:
            self.wandb_project = None
        logger.info("TrainingManager initialized for model %s (backend=%s)", model_name, backend)

    def train(self, train_dataset, eval_dataset=None, output_dir: str = "./trained_model", **training_args_kwargs: Any) -> None:
        """
        Fine-tune the model on provided datasets.

        Args:
            train_dataset: Dataset for training.
            eval_dataset: Optional dataset for evaluation.
            output_dir: Directory to save model.
            training_args_kwargs: Keyword args for TrainingArguments (e.g., num_train_epochs, per_device_train_batch_size).
        """
        # Auto-config GPU mixed-precision for PyTorch backend
        if self.handler.backend == "pt":
            try:
                import torch

                if torch.cuda.is_available():
                    training_args_kwargs.setdefault("fp16", True)
            except ImportError:
                pass
        # Choose training framework based on backend
        if self.handler.backend == "jax":
            from transformers import FlaxTrainer, FlaxTrainingArguments

            args = FlaxTrainingArguments(output_dir=output_dir, evaluation_strategy="steps" if eval_dataset is not None else "no", logging_dir=os.path.join(output_dir, "logs"), **training_args_kwargs)
            trainer = FlaxTrainer(
                model=self.handler.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.handler.tokenizer,
            )
        else:
            from transformers import Trainer, TrainingArguments

            args = TrainingArguments(output_dir=output_dir, evaluation_strategy="steps" if eval_dataset is not None else "no", logging_dir=os.path.join(output_dir, "logs"), **training_args_kwargs)
            trainer = Trainer(
                model=self.handler.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.handler.tokenizer,
            )
        # Optional W&B integration
        if self.wandb_project:
            trainer.add_callback(WandbCallback)
        trainer.train()
        trainer.save_model(output_dir)
        self.handler.tokenizer.save_pretrained(output_dir)
        if self.wandb_project:
            wandb.finish()
        logger.info("Training complete; model saved to %s", output_dir)

    def hyperparameter_search(self, train_dataset, eval_dataset, output_dir: str = "./hyperopt_model", n_trials: int = 10, direction: str = "minimize", **training_args_base: Any) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
            output_dir: Base output directory.
            n_trials: Number of Optuna trials.
            direction: "minimize" for loss, "maximize" for metric.
            training_args_base: Base TrainingArguments kwargs.

        Returns:
            Best hyperparameters from the study.
        """

        def objective(trial):
            # Define search space
            trial_args = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            }
            args = TrainingArguments(output_dir=output_dir, evaluation_strategy="steps", logging_dir=os.path.join(output_dir, "logs"), **training_args_base, **trial_args)
            trainer = Trainer(
                model=self.handler.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.handler.tokenizer,
            )
            metrics = trainer.evaluate()
            # Objective: minimize eval_loss
            return metrics.get("eval_loss", float("inf"))

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logger.info("Optuna best hyperparameters: %s", best_params)
        return best_params

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from lylex.training import TrainingManager
from plugin_utils.logging import get_logger
from plugin_utils.metrics import metrics
from plugin_utils.validation import require_json_fields

# HTTP API for machine learning tasks
ml_bp = Blueprint("ml_tasks_api", __name__, url_prefix="/api/ml")
logger = get_logger(__name__)


@ml_bp.route("/train", methods=["POST"])
@jwt_required()
@require_json_fields("model_name", "train_dataset")
@metrics
def train_api():
    """
    Train a model via HTTP API, using a JSON configuration.
    Expects 'model_name', 'train_dataset', and optionally 'eval_dataset',
    'output_dir', and 'training_args' in the JSON payload.
    """
    cfg = request.get_json(force=True)
    # Validation for model_name and train_dataset is handled by @require_json_fields
    logger.info(f"Received training request for model: {cfg.get('model_name')}")
    try:
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
        logger.info(
            f"Training complete for model: {cfg.get('model_name')}. Output: {cfg.get('output_dir', './trained_model')}"
        )
        return jsonify(
            {
                "message": f"Training started. Model will be saved to {cfg.get('output_dir', './trained_model')}"
            }
        )
    except Exception as e:
        logger.error(
            f"ML training error for model {cfg.get('model_name')}: {e}", exc_info=True
        )
        raise


@ml_bp.route("/tune", methods=["POST"])
@jwt_required()
@require_json_fields("model_name", "train_dataset", "eval_dataset", "study_name")
@metrics
def tune_api():
    """
    Tune hyperparameters for a model via HTTP API, using a JSON configuration.
    Expects 'model_name', 'train_dataset', 'eval_dataset', and 'study_name'.
    Optionally 'n_trials', 'output_dir', 'storage', 'metric_name'.
    """
    cfg = request.get_json(force=True)
    # Validation for required fields is handled by @require_json_fields
    logger.info(
        f"Received tuning request for model: {cfg.get('model_name')}, study: {cfg.get('study_name')}"
    )
    try:
        tm = TrainingManager(
            model_name=cfg["model_name"],
            backend=cfg.get("backend", "pt"),
            wandb_project=cfg.get("wandb_project"),
        )
        study = tm.tune_hyperparameters(
            train_dataset_path=cfg["train_dataset"],
            eval_dataset_path=cfg["eval_dataset"],
            study_name=cfg["study_name"],
            n_trials=cfg.get("n_trials", 10),
            output_dir=cfg.get("output_dir", "./tuned_model"),
            storage=cfg.get("storage"),  # e.g., 'sqlite:///tuning.db'
            metric_name=cfg.get("metric_name", "eval_loss"),
        )
        logger.info(
            f"Tuning complete for model: {cfg.get('model_name')}, study: {study.study_name}. Best trial: {study.best_trial.value}"
        )
        return jsonify(
            {
                "message": f"Hyperparameter tuning complete for study '{study.study_name}'.",
                "best_trial": {
                    "value": study.best_trial.value,
                    "params": study.best_trial.params,
                },
            }
        )
    except Exception as e:
        logger.error(
            f"ML tuning error for model {cfg.get('model_name')}, study {cfg.get('study_name')}: {e}",
            exc_info=True,
        )
        raise

import os
import json
import logging
import time
from typing import Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import Counter
import numpy as np

from datasets import Dataset

from .ai import LylexAgent
from .db import LylexDB
from .training import TrainingManager
from .neuron import Neuron

logger = logging.getLogger(__name__)
# Metrics for observability
_cycles_counter = Counter('brain_learning_cycles_total', 'Total autonomous learning cycles')
_new_interactions_counter = Counter('brain_new_interactions_total', 'Total new interactions learned by Brain')
_errors_counter = Counter('brain_learning_errors_total', 'Total errors during Brain learning')

class Brain:
    """
    Autonomous self-learning module for Lylex that periodically fine-tunes
    the model on stored conversation interactions without explicit prompting.
    """
    def __init__(
        self,
        model_name: str,
        backend: str = "pt",
        memory_db: Optional[LylexDB] = None,
        memory_limit: int = 100,
        interval_minutes: int = 60,
        train_epochs: int = 1,
        batch_size: int = 1,
        mixed_precision: bool = True,
        peft_r: int = 8,
        peft_alpha: int = 16,
        peft_dropout: float = 0.1,
        wandb_project: Optional[str] = None,
        neuron_hoc_file: Optional[str] = None,
        neuron_dt: Optional[float] = None,
        neuron_tstop: Optional[float] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        # Initialize memory DB and training parameters
        self.memory_db = memory_db or LylexDB(vector_dimension=memory_db.vector_dimension if memory_db else 384)
        self.memory_limit = memory_limit
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        # Track last processed interaction ID for incremental training
        self.last_id = self._load_last_id()
        # Initialize agent and training manager with LoRA settings
        self.agent = LylexAgent(model_name=model_name, backend=backend, memory_db=self.memory_db, memory_limit=memory_limit)
        self.training_manager = TrainingManager(
            model_name,
            backend,
            peft_r=peft_r,
            peft_alpha=peft_alpha,
            peft_dropout=peft_dropout,
            wandb_project=wandb_project
        )
        # Initialize NEURON simulator if provided
        self.simulator = Neuron(hoc_file=neuron_hoc_file) if neuron_hoc_file else None
        # Simulation parameters
        self.sim_dt = neuron_dt
        self.sim_tstop = neuron_tstop
        if self.simulator:
            if neuron_dt is not None:
                self.simulator.set_dt(neuron_dt)
            if neuron_tstop is not None:
                self.simulator.set_tstop(neuron_tstop)
        # Schedule autonomous learning via APScheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.learn,
            'interval',
            minutes=interval_minutes,
            id=f'brain_{model_name}',
            replace_existing=True
        )
        self.scheduler.start()
        logger.info(f"Brain initialized: autonomous learning every {interval_minutes} minutes.")

    def _extract_trace_features(self, t: np.ndarray, v: np.ndarray) -> dict:
        """Extract basic features from a simulation trace."""
        if v is None or len(v) == 0:
            return {
                "mean_value": None,
                "max_value": None,
                "min_value": None,
                "duration_ms": float(t[-1] - t[0]) if t is not None and len(t) > 1 else 0,
                "num_points": 0,
            }
        return {
            "mean_value": float(np.mean(v)),
            "max_value": float(np.max(v)),
            "min_value": float(np.min(v)),
            "duration_ms": float(t[-1] - t[0]) if t is not None and len(t) > 1 else 0,
            "num_points": len(v),
        }

    def learn(self):  # called by APScheduler
        """
        Perform a learning cycle: fetch stored interactions and fine-tune the model.
        """
        _cycles_counter.inc()
        try:
            # Export all interactions and filter new ones
            raw = self.memory_db.export_interactions(limit=None)
            new_entries = [e for e in raw if e['id'] > self.last_id]
            if not new_entries:
                logger.info("Brain: no new interactions for learning.")
                return
            new_entries.sort(key=lambda e: e['id'])
            texts = [f"{e['metadata'].get('prompt', '')}\n{e['metadata'].get('response', '')}" for e in new_entries]
            dataset = Dataset.from_dict({'text': texts})
            # Fine-tune model on new interactions
            self.training_manager.train(
                train_dataset=dataset,
                output_dir=f"./brain_trained_{self.model_name}",
                num_train_epochs=self.train_epochs,
                per_device_train_batch_size=self.batch_size,
                fp16=self.mixed_precision
            )
            count = len(texts)
            _new_interactions_counter.inc(count)
            # Update checkpoint
            self.last_id = new_entries[-1]['id']
            self._save_last_id()
            # Run NEURON simulation and store trace
            if self.simulator:
                try:
                    t, v = self.simulator.record_trace()
                    features = self._extract_trace_features(t, v)
                    sim_interaction_data = {
                        'trace': {'time': t.tolist(), 'values': v.tolist()},
                        'features': features
                    }
                    self.memory_db.store_interaction(
                        prompt='neuron_simulation_cycle',
                        response=json.dumps(sim_interaction_data),
                        metadata={'source': 'brain_learn_cycle', 'dt': self.sim_dt, 'tstop': self.sim_tstop}
                    )
                    logger.info('Brain: stored neuron simulation trace with features.')
                except Exception as sim_e:
                    logger.error(f'Brain simulation error: {sim_e}')
            logger.info(f"Brain: learned from {count} new interactions.")
        except Exception as e:
            _errors_counter.inc()
            logger.exception(f"Brain learning error: {e}")

    def stop(self):
        """
        Stop the autonomous learning scheduler.
        """
        self.scheduler.remove_job(f'brain_{self.model_name}')
        self.scheduler.shutdown(wait=False)
        logger.info("Brain autonomous learning stopped.")

    def _load_last_id(self):
        checkpoint = f'.brain_checkpoint_{self.model_name}.json'
        try:
            with open(checkpoint, 'r') as f:
                return json.load(f).get('last_id', 0)
        except Exception:
            return 0

    def _save_last_id(self):
        checkpoint = f'.brain_checkpoint_{self.model_name}.json'
        try:
            with open(checkpoint, 'w') as f:
                json.dump({'last_id': self.last_id}, f)
        except Exception as e:
            logger.error(f"Failed to save Brain checkpoint: {e}")

    def simulate_sweep(self,
                       param_name: str,
                       values: List[float],
                       param_name2: Optional[str] = None,
                       values2: Optional[List[float]] = None,
                       ref=None,
                       t_ref=None):
        """
        Sweep one or two NEURON simulation parameters and record a trace at each combination.
        Stores each trace into the memory DB and returns a dict mapping value(s)->(t_array, v_array).
        """
        if not self.simulator:
            raise RuntimeError("No NEURON simulator configured in Brain")

        results = self.simulator.sweep_parameter(
            param_name, values, param_name2, values2, ref=ref, t_ref=t_ref
        )

        if param_name2 and values2: # 2D sweep
            for val1, nested_results in results.items():
                for val2, (t, v) in nested_results.items():
                    features = self._extract_trace_features(t, v)
                    interaction_data = {
                        'trace': {'time': t.tolist(), 'values': v.tolist()},
                        'features': features
                    }
                    prompt = f'neuron_sweep_2D_{param_name}_{val1}_{param_name2}_{val2}'
                    metadata = {
                        'source': 'brain_simulate_sweep_2D',
                        'param1_name': param_name, 'param1_value': val1,
                        'param2_name': param_name2, 'param2_value': val2,
                        'dt': self.sim_dt, 'tstop': self.sim_tstop
                    }
                    self.memory_db.store_interaction(
                        prompt=prompt,
                        response=json.dumps(interaction_data),
                        metadata=metadata
                    )
        else: # 1D sweep
            for val, (t, v) in results.items():
                features = self._extract_trace_features(t, v)
                interaction_data = {
                    'trace': {'time': t.tolist(), 'values': v.tolist()},
                    'features': features
                }
                prompt = f'neuron_sweep_1D_{param_name}_{val}'
                metadata = {
                    'source': 'brain_simulate_sweep_1D',
                    'param_name': param_name, 'param_value': val,
                    'dt': self.sim_dt, 'tstop': self.sim_tstop
                }
                self.memory_db.store_interaction(
                    prompt=prompt,
                    response=json.dumps(interaction_data),
                    metadata=metadata
                )
        return results 
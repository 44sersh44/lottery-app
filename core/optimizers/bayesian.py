import optuna
import json
import os
from .base import BaseOptimizer

class BayesianOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, n_trials=50):
        tuner = self.tuner
        total_blocks = len(all_blocks)

        def objective(trial):
            params = {}
            for p_name, p_values in param_grid.items():
                if not p_values:
                    continue
                if p_name in tuner.PARAM_TYPES and tuner.PARAM_TYPES[p_name] in ('integer', 'small_int'):
                    if len(p_values) > 1:
                        min_val = min(p_values)
                        max_val = max(p_values)
                    else:
                        min_val = 1
                        max_val = 10
                    params[p_name] = trial.suggest_int(p_name, min_val, max_val)
                else:
                    if all(isinstance(v, (int, float)) for v in p_values):
                        min_val = min(p_values)
                        max_val = max(p_values)
                        params[p_name] = trial.suggest_float(p_name, min_val, max_val)
                    else:
                        params[p_name] = trial.suggest_categorical(p_name, p_values)
            if model_name == "Markov" and "order" in params and isinstance(params["order"], int):
                params["order"] = (params["order"], 1, 0)
            score, _, _, _, _ = tuner._walk_forward_test(
                model_name, params, all_blocks, initial_window,
                pick_count, total_numbers, field
            )
            return -score  # Optuna минимизирует

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        if model_name == "Markov" and "order" in best_params and isinstance(best_params["order"], int):
            best_params["order"] = (best_params["order"], 1, 0)
        best_score = -study.best_value
        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_avg_matches": 0,
            "best_total_matches": 0,
            "matches_list": [],
            "total_iterations": total_blocks - initial_window,
            "iterations_file": None
        }
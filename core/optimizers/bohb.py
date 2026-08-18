import optuna
from .base import BaseOptimizer
from .hyperband import HyperbandOptimizer

class BOHBOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, n_trials=50):
        tuner = self.tuner
        print(f"  [BOHB] Запуск BOHB с {n_trials} пробами...")
        try:
            from optuna.integration import BOHB
            print("  [INFO] BOHB импортирован успешно (optuna.integration)")
        except ImportError:
            try:
                from optuna.integration.bohb import BOHB
                print("  [INFO] BOHB импортирован успешно (optuna.integration.bohb)")
            except ImportError:
                print("  [WARN] BOHB не найден, используем Hyperband")
                fallback = HyperbandOptimizer(tuner)
                return fallback.optimize(model_name, param_grid, all_blocks, initial_window,
                                         pick_count, total_numbers, field)

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
                pick_count, total_numbers, field,
                max_iterations=20
            )
            return score

        try:
            if hasattr(BOHB, 'BOHBSampler'):
                sampler = BOHB.BOHBSampler(seed=42, n_startup_trials=10)
            else:
                sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        except Exception as e:
            print(f"  [ERR] Ошибка BOHB: {e}, переключение на Hyperband")
            fallback = HyperbandOptimizer(tuner)
            return fallback.optimize(model_name, param_grid, all_blocks, initial_window,
                                     pick_count, total_numbers, field)

        best_params = study.best_params
        if model_name == "Markov" and "order" in best_params and isinstance(best_params["order"], int):
            best_params["order"] = (best_params["order"], 1, 0)

        final_score, final_iterations, final_avg, final_total, final_matches = tuner._walk_forward_test(
            model_name, best_params, all_blocks, initial_window,
            pick_count, total_numbers, field
        )
        return {
            "best_params": best_params,
            "best_score": final_score,
            "best_avg_matches": final_avg,
            "best_total_matches": sum(final_matches),
            "matches_list": final_matches,
            "total_iterations": len(final_iterations),
            "iterations_file": None
        }
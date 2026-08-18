import optuna
import json
import os
from .base import BaseOptimizer

class HybridOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, start_params=None, n_trials=30):
        tuner = self.tuner
        total_blocks = len(all_blocks)
        param_names = list(param_grid.keys())
        if not param_names:
            score, iterations, avg_matches, total_matches, matches_list = tuner._walk_forward_test(
                model_name, {}, all_blocks, initial_window,
                pick_count, total_numbers, field
            )
            return {
                "best_params": {},
                "best_score": score,
                "best_avg_matches": avg_matches,
                "best_total_matches": sum(matches_list),
                "matches_list": matches_list,
                "total_iterations": len(iterations),
                "iterations_file": None
            }

        # 1. Адаптивный coarse
        from .adaptive import AdaptiveOptimizer
        coarse_optimizer = AdaptiveOptimizer(tuner)
        coarse_result = coarse_optimizer.optimize(
            model_name, param_grid, all_blocks, initial_window,
            pick_count, total_numbers, field,
            start_params=start_params or {},
            levels=["coarse"]
        )
        best_coarse_params = coarse_result["best_params"]
        coarse_score = coarse_result["best_score"]
        print(f"  [HYBRID] Coarse лучший: {best_coarse_params} (score={coarse_score:.3f})")

        # 2. Сужаем диапазон
        search_ranges = {}
        for p in param_names:
            if p in param_grid and param_grid[p] and not all(isinstance(x, (int, float)) for x in param_grid[p]):
                search_ranges[p] = param_grid[p]
            else:
                best_val = best_coarse_params.get(p, 0)
                if p == "order" and isinstance(best_val, tuple):
                    best_val = best_val[0]
                step = tuner.PARAM_SPECIFIC_STEPS.get(p, {}).get("coarse", 1)
                min_val = best_val - 2 * step
                max_val = best_val + 2 * step
                global_min, global_max = tuner.PARAM_LIMITS.get(p, (None, None))
                if global_min is not None:
                    min_val = max(min_val, global_min)
                if global_max is not None:
                    max_val = min(max_val, global_max)
                if min_val > max_val:
                    mid = (min_val + max_val) / 2 if min_val + max_val > 0 else best_val
                    step_adj = abs(max_val - min_val) / 2 + 1
                    min_val = max(mid - step_adj, global_min if global_min is not None else 0)
                    max_val = min(mid + step_adj, global_max if global_max is not None else 100)
                    if min_val > max_val:
                        min_val = max(best_val - step, 0)
                        max_val = min(best_val + step, 100)
                        if min_val > max_val:
                            min_val = max_val = best_val
                if p in tuner.PARAM_TYPES and tuner.PARAM_TYPES[p] in ("integer", "small_int"):
                    min_val = int(round(min_val))
                    max_val = int(round(max_val))
                search_ranges[p] = (min_val, max_val)

        # 3. Байесовское уточнение
        def objective(trial):
            params = {}
            for p_name, p_range in search_ranges.items():
                if isinstance(p_range, (list, tuple)) and len(p_range) == 2 and all(isinstance(x, (int, float)) for x in p_range):
                    if p_name in tuner.PARAM_TYPES and tuner.PARAM_TYPES[p_name] in ("integer", "small_int"):
                        params[p_name] = trial.suggest_int(p_name, int(p_range[0]), int(p_range[1]))
                    else:
                        params[p_name] = trial.suggest_float(p_name, p_range[0], p_range[1])
                else:
                    params[p_name] = trial.suggest_categorical(p_name, p_range)
            if model_name == "Markov" and "order" in params and isinstance(params["order"], int):
                params["order"] = (params["order"], 1, 0)
            score, _, _, _, _ = tuner._walk_forward_test(
                model_name, params, all_blocks, initial_window,
                pick_count, total_numbers, field,
                max_iterations=20
            )
            return score

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        if model_name == "Markov" and "order" in best_params and isinstance(best_params["order"], int):
            best_params["order"] = (best_params["order"], 1, 0)

        final_score, final_iterations, final_avg, final_total, final_matches = tuner._walk_forward_test(
            model_name, best_params, all_blocks, initial_window,
            pick_count, total_numbers, field
        )
        print(f"  [HYBRID] Итоговый лучший: {best_params} (score={final_score:.3f}, avg={final_avg:.2f})")
        return {
            "best_params": best_params,
            "best_score": final_score,
            "best_avg_matches": final_avg,
            "best_total_matches": sum(final_matches),
            "matches_list": final_matches,
            "total_iterations": len(final_iterations),
            "iterations_file": None
        }
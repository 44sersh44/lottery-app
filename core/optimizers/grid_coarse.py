import itertools
from .base import BaseOptimizer

class GridCoarseOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, **kwargs):
        tuner = self.tuner
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        if not param_values:
            score, iterations, _, _, _ = tuner._walk_forward_test(
                model_name, {}, all_blocks, initial_window, pick_count, total_numbers, field
            )
            return {"best_params": {}, "best_score": score, "best_avg_matches": 0,
                    "matches_list": [], "total_iterations": len(iterations), "iterations_file": None}

        combos = list(itertools.product(*param_values))
        best_score = -1
        best_params = None
        for combo in combos:
            params = dict(zip(param_names, combo))
            score, _, _, _, _ = tuner._walk_forward_test(
                model_name, params, all_blocks, initial_window, pick_count, total_numbers, field
            )
            if score > best_score:
                best_score = score
                best_params = params
        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_avg_matches": 0,
            "matches_list": [],
            "total_iterations": 0,
            "iterations_file": None
        }
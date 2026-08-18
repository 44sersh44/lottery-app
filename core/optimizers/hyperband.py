import random
import math
from .base import BaseOptimizer

class HyperbandOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, max_iter=100, eta=3):
        tuner = self.tuner
        print(f"  [HYPERBAND] Запуск для {model_name} с max_iter={max_iter}, eta={eta}")

        def sample_params():
            params = {}
            for p_name, p_values in param_grid.items():
                if not p_values:
                    params[p_name] = None
                else:
                    params[p_name] = random.choice(p_values)
            return params

        def run_with_budget(params, budget):
            max_iterations = max(1, int(budget * (len(all_blocks) - initial_window)))
            score, _, _, _, _ = tuner._walk_forward_test(
                model_name, params, all_blocks, initial_window,
                pick_count, total_numbers, field,
                max_iterations=max_iterations
            )
            return score

        s_max = int(math.log(max_iter, eta))
        B = max_iter
        best_score = -1
        best_params = None

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil((B / (s + 1)) * (eta ** s)))
            r = B / (eta ** s)
            candidates = [sample_params() for _ in range(n)]
            scores = [run_with_budget(p, r / max_iter) for p in candidates]
            sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            n_selected = max(1, n // eta)
            candidates = [candidates[i] for i in sorted_idx[:n_selected]]
            while n_selected > 1:
                r = r * eta
                n_selected = max(1, n_selected // eta)
                scores = [run_with_budget(p, r / max_iter) for p in candidates]
                sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                candidates = [candidates[i] for i in sorted_idx[:n_selected]]
            if candidates:
                final_score = run_with_budget(candidates[0], 1.0)
                if final_score > best_score:
                    best_score = final_score
                    best_params = candidates[0]

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
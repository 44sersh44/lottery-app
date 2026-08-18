import json
import os
import itertools
from .base import BaseOptimizer

class AdaptiveOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field, start_params=None, levels=None):
        tuner = self.tuner
        param_names = list(param_grid.keys())
        if not param_names:
            score, iterations, avg_matches, total_matches, matches_list = tuner._walk_forward_test(
                model_name, {}, all_blocks, initial_window,
                pick_count, total_numbers, field
            )
            iter_filename = f"wf_iter_{model_name}_field{field}.json"
            best_iterations_file = os.path.join(tuner.profile_path, iter_filename)
            with open(best_iterations_file, 'w', encoding='utf-8') as f:
                json.dump(iterations, f, indent=2, ensure_ascii=False)
            return {
                "best_params": {},
                "best_score": score,
                "best_avg_matches": avg_matches,
                "best_total_matches": sum(matches_list),
                "matches_list": matches_list,
                "total_iterations": len(iterations),
                "iterations_file": best_iterations_file
            }

        # Настройка уровней
        if levels is None:
            heavy_models = ["LSTM", "Transformer", "Prophet"]
            if model_name in heavy_models:
                levels = ["coarse", "medium"]
            else:
                levels = ["coarse", "medium", "fine", "ultra"]

        points_per_level = {"coarse": 5, "medium": 3, "fine": 3, "ultra": 3}
        heavy_models = ["LSTM", "Transformer", "Prophet"]
        max_wf_iterations = 20 if model_name in heavy_models else None

        # Корректировка сетки для тяжёлых
        if model_name in heavy_models:
            if model_name == "LSTM":
                if "lookback" in param_grid:
                    param_grid["lookback"] = [5, 10]
                if "epochs" in param_grid:
                    param_grid["epochs"] = [10, 20]
            elif model_name == "Transformer":
                if "d_model" in param_grid:
                    param_grid["d_model"] = [32, 64]
                if "n_heads" in param_grid:
                    param_grid["n_heads"] = [4, 8]
            # Prophet оставляем как есть

        best_params = start_params.copy() if start_params else {}
        best_score = -1
        best_avg = 0
        best_matches_list = []
        best_iterations_file = None
        total_blocks = len(all_blocks)

        # Определяем начальные диапазоны и шаги
        global_ranges = {}
        global_steps = {}
        for p in param_names:
            if p in param_grid and param_grid[p] and all(isinstance(x, (int, float)) for x in param_grid[p]):
                min_val = min(param_grid[p])
                max_val = max(param_grid[p])
                if len(param_grid[p]) > 1:
                    step = param_grid[p][1] - param_grid[p][0]
                else:
                    step = tuner.PARAM_SPECIFIC_STEPS.get(p, {}).get("coarse", 1)
            else:
                limits = tuner.PARAM_LIMITS.get(p, (None, None))
                min_val, max_val = limits
                if min_val is None or max_val is None:
                    min_val, max_val = 0, 100
                step = tuner.PARAM_SPECIFIC_STEPS.get(p, {}).get("coarse", 1)
            global_ranges[p] = (min_val, max_val)
            global_steps[p] = step

        current_ranges = global_ranges.copy()
        current_steps = global_steps.copy()

        def is_better(score, avg_matches, params, best_score, best_avg, best_params):
            if score > best_score:
                return True
            if score == best_score and avg_matches > best_avg:
                return True
            if score == best_score and avg_matches == best_avg:
                comp = sum(v if isinstance(v, (int, float)) else 0 for v in params.values())
                best_comp = sum(v if isinstance(v, (int, float)) else 0 for v in best_params.values())
                return comp < best_comp
            return False

        for idx, level in enumerate(levels):
            print(f"\n  === УРОВЕНЬ {level.upper()} ===")
            if idx == 0:
                combos = tuner._generate_adaptive_combinations(param_names, param_grid, best_params, level, current_ranges)
            else:
                combo_values = []
                points_count = points_per_level.get(level, 3)
                for p in param_names:
                    if p == "order" and param_grid.get(p) and any(isinstance(v, tuple) for v in param_grid[p]):
                        if p in best_params:
                            fixed_val = best_params[p]
                        else:
                            fixed_val = param_grid[p][0] if param_grid[p] else None
                        combo_values.append([fixed_val])
                        continue
                    if p in param_grid and param_grid[p] and not all(isinstance(x, (int, float)) for x in param_grid[p]):
                        combo_values.append(param_grid[p])
                    else:
                        best_val = best_params.get(p, 0)
                        if p == "order" and isinstance(best_val, tuple):
                            best_val = best_val[0]
                        step = current_steps.get(p, 1)
                        half = points_count // 2
                        vals = []
                        for i in range(-half, half + 1):
                            vals.append(best_val + i * step)
                        if p in tuner.PARAM_TYPES and tuner.PARAM_TYPES[p] in ("integer", "small_int"):
                            vals = [int(round(v)) for v in vals]
                        min_val, max_val = global_ranges.get(p, (None, None))
                        if min_val is not None:
                            vals = [max(v, min_val) for v in vals]
                        if max_val is not None:
                            vals = [min(v, max_val) for v in vals]
                        vals = sorted(set(vals))
                        combo_values.append(vals)
                combos = list(itertools.product(*combo_values))

            print(f"    Проверка {len(combos)} комбинаций...")
            level_best_score = -1
            level_best_params = None
            level_best_avg = 0
            level_best_matches = []
            level_best_iter_file = None

            for combo in combos:
                params = dict(zip(param_names, combo))
                if model_name == "Markov" and "order" in params and isinstance(params["order"], int):
                    params["order"] = (params["order"], 1, 0)
                score, iterations, avg_matches, total_matches, matches_list = tuner._walk_forward_test(
                    model_name, params, all_blocks, initial_window,
                    pick_count, total_numbers, field,
                    max_iterations=max_wf_iterations
                )
                total_correct = sum(1 for it in iterations if it.get("correct", False))
                total_tests = len(iterations)
                status = "✅" if is_better(score, avg_matches, params, level_best_score, level_best_avg, level_best_params) else "  "
                if field == 1:
                    print(f"    {status} {params} → угадал: {score:.0%} ({total_correct}/{total_tests}), ср.совпадений: {avg_matches:.2f}")
                else:
                    print(f"    {status} {params} → {score:.3f} ({total_correct}/{total_tests})")
                if is_better(score, avg_matches, params, level_best_score, level_best_avg, level_best_params):
                    level_best_score = score
                    level_best_params = params
                    level_best_avg = avg_matches
                    level_best_matches = matches_list
                    iter_filename = f"wf_iter_{model_name}_field{field}_level{level}.json"
                    level_best_iter_file = os.path.join(tuner.profile_path, iter_filename)
                    with open(level_best_iter_file, 'w', encoding='utf-8') as f:
                        json.dump(iterations, f, indent=2, ensure_ascii=False)

            if is_better(level_best_score, level_best_avg, level_best_params, best_score, best_avg, best_params):
                best_score = level_best_score
                best_params = level_best_params
                best_avg = level_best_avg
                best_matches_list = level_best_matches
                best_iterations_file = level_best_iter_file
                print(f"    ✅ Лучшее на уровне {level}: {best_params} (score={best_score:.3f}, avg={best_avg:.3f})")
            else:
                print(f"    ❌ Улучшения на уровне {level} нет, продолжаем с предыдущими параметрами")

            if idx < len(levels) - 1:
                for p in param_names:
                    if p in global_steps:
                        if p in tuner.PARAM_SPECIFIC_STEPS:
                            next_step = tuner.PARAM_SPECIFIC_STEPS[p].get(levels[idx+1], global_steps[p])
                        else:
                            param_type = tuner.PARAM_TYPES.get(p, "integer")
                            next_step = tuner.ADAPTIVE_STEPS.get(param_type, {}).get(levels[idx+1], global_steps[p] / 2)
                        current_steps[p] = next_step

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_avg_matches": best_avg,
            "best_total_matches": sum(best_matches_list) if best_matches_list else 0,
            "matches_list": best_matches_list,
            "total_iterations": total_blocks - initial_window,
            "iterations_file": best_iterations_file
        }
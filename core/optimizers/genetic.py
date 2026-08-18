import random
from .base import BaseOptimizer

class GeneticOptimizer(BaseOptimizer):
    def optimize(self, model_name, param_grid, all_blocks, initial_window,
                 pick_count, total_numbers, field,
                 population_size=20, generations=10, mutation_rate=0.1):
        tuner = self.tuner

        def random_individual():
            ind = {}
            for p_name, p_values in param_grid.items():
                if not p_values:
                    ind[p_name] = None
                else:
                    ind[p_name] = random.choice(p_values)
            return ind

        def evaluate(individual):
            clean = {k: v for k, v in individual.items() if v is not None}
            score, _, _, _, _ = tuner._walk_forward_test(
                model_name, clean, all_blocks, initial_window,
                pick_count, total_numbers, field
            )
            return score

        def crossover(parent1, parent2):
            child = {}
            for p in param_grid.keys():
                if random.random() < 0.5:
                    child[p] = parent1.get(p)
                else:
                    child[p] = parent2.get(p)
            return child

        def mutate(individual):
            for p in param_grid.keys():
                if p in param_grid and param_grid[p] and random.random() < mutation_rate:
                    individual[p] = random.choice(param_grid[p])
            return individual

        population = [random_individual() for _ in range(population_size)]
        best_ever_score = -1
        best_ever_individual = None

        for gen in range(generations):
            scores = [(evaluate(ind), ind) for ind in population]
            scores.sort(key=lambda x: x[0], reverse=True)
            if scores[0][0] > best_ever_score:
                best_ever_score = scores[0][0]
                best_ever_individual = scores[0][1]
            elite_size = max(2, population_size // 4)
            elites = scores[:elite_size]
            new_population = [ind for _, ind in elites]
            while len(new_population) < population_size:
                parent1 = random.choice(elites)[1]
                parent2 = random.choice(elites)[1]
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            population = new_population
            print(f"  [GEN] Поколение {gen+1}/{generations}, лучший score: {best_ever_score:.3f}")

        if best_ever_individual is None:
            best_ever_individual = max(population, key=evaluate)

        final_score, final_iterations, final_avg, final_total, final_matches = tuner._walk_forward_test(
            model_name, best_ever_individual, all_blocks, initial_window,
            pick_count, total_numbers, field
        )
        return {
            "best_params": best_ever_individual,
            "best_score": final_score,
            "best_avg_matches": final_avg,
            "best_total_matches": sum(final_matches),
            "matches_list": final_matches,
            "total_iterations": len(final_iterations),
            "iterations_file": None
        }
from .adaptive import AdaptiveOptimizer
from .bayesian import BayesianOptimizer
from .hybrid import HybridOptimizer
from .hyperband import HyperbandOptimizer
from .genetic import GeneticOptimizer
from .bohb import BOHBOptimizer
from .grid_coarse import GridCoarseOptimizer

OPTIMIZER_REGISTRY = {
    'adaptive': AdaptiveOptimizer,
    'bayesian': BayesianOptimizer,
    'hybrid': HybridOptimizer,
    'hyperband': HyperbandOptimizer,
    'genetic': GeneticOptimizer,
    'bohb': BOHBOptimizer,
    'grid_coarse': GridCoarseOptimizer,
}
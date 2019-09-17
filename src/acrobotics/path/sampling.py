import enum
from acrolib.sampling import SampleMethod


class SearchStrategy(enum.Enum):
    """
    Sampling based planners can use different strategies
    to convert a cartesian path to joint space.
    """

    GRID = 0
    INCREMENTAL = 1
    MIN_INCREMENTAL = 2


class SamplingSetting:
    """
    Settings for the sampling based planners.
    """

    def __init__(
        self,
        search_strategy: SearchStrategy,
        iterations: int = None,
        sample_method: SampleMethod = None,
        num_samples: int = None,
        desired_num_samples: int = None,
        max_search_iters: int = None,
        tolerance_reduction_factor: float = None,
        cost_function: callable = None,
        weights=None,
        use_state_cost=False,
        state_cost_weight=1.0,
    ):
        self.cost_function = cost_function
        self.weights = weights
        self.use_state_cost = use_state_cost
        self.state_cost_weight = state_cost_weight

        if iterations is None or iterations == 1:
            self.iterations = 1
        else:
            self.iterations = iterations
            assert tolerance_reduction_factor is not None
            self.tolerance_reduction_factor = tolerance_reduction_factor

        self.search_strategy = search_strategy
        if self.search_strategy == SearchStrategy.GRID:
            pass
        elif self.search_strategy == SearchStrategy.INCREMENTAL:
            assert sample_method is not None
            assert num_samples is not None
            self.sample_method = sample_method
            self.num_samples = num_samples
        elif self.search_strategy == SearchStrategy.MIN_INCREMENTAL:
            assert sample_method is not None
            assert desired_num_samples is not None
            assert max_search_iters is not None
            self.sample_method = sample_method
            self.desired_num_samples = desired_num_samples
            self.max_search_iters = max_search_iters
            self.step_size = 1

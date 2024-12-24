import numpy as np
from scipy import interpolate
from scipy.stats import norm

class MultiLinearTransformer(object):
    def __init__(
        self,
        target_min_bound: float = 0.2,
        target_max_bound: float = 0.8,
        linear_num: int = 100,
    ) -> None:
        self.target_min_bound = target_min_bound
        self.target_max_bound = target_max_bound
        self.linear_num = linear_num

        self.source_values = []
        self.target_values = []
        self.linear_func = None
        self.inv_linear_func = None

        self.updateBounds(target_min_bound, target_max_bound, linear_num)
        return

    def reset(self) -> bool:
        self.source_values = []
        self.target_values = []
        self.linear_func = None
        self.inv_linear_func = None
        return True

    def updateBounds(
        self,
        target_min_bound: float = 0.2,
        target_max_bound: float = 0.8,
        linear_num: int = 10,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> bool:
        self.reset()

        self.target_min_bound = target_min_bound
        self.target_max_bound = target_max_bound
        self.linear_num = linear_num

        target_bounds = np.linspace(self.target_min_bound, self.target_max_bound, linear_num + 1, dtype=np.float64)

        self.target_values = [norm.ppf(target_bound, loc=mean, scale=std) for target_bound in target_bounds]
        return True

    def fit(self, data: np.ndarray) -> bool:
        self.source_values = [np.percentile(data, 100.0 * i / self.linear_num) for i in range(self.linear_num + 1)]

        self.linear_func = interpolate.interp1d(self.source_values, self.target_values)
        self.inv_linear_func = interpolate.interp1d(self.target_values, self.source_values)
        return True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.linear_func is None:
            print('[WARN][MultiLinearTransformer::transform]')
            print('\t linear function not exist! will return source data!')
            return data

        return self.linear_func(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.inv_linear_func is None:
            print('[WARN][MultiLinearTransformer::transform]')
            print('\t inv linear function not exist! will return source data!')
            return data

        return self.inv_linear_func(data)

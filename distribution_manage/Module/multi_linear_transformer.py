import numpy as np

class MultiLinearTransformer(object):
    def __init__(
        self,
        linear_bounds: list = [0.25, 0.5, 0.75],
    ) -> None:
        self.linear_bounds = linear_bounds

        self.linear_bound_values = []

        self.updateLinearBounds()
        return

    def updateLinearBounds(self) -> bool:
        if len(self.linear_bounds) == 0:
            self.linear_bounds = [0.0, 1.0]
            return True

        if self.linear_bounds[0] != 0.0:
            self.linear_bounds.insert(0, 0.0)

        if self.linear_bounds[-1] != 1.0:
            self.linear_bounds.append(1.0)
        return True

    def fit(self, data: np.ndarray) -> bool:
        q_list = [np.percentile(data, i * 100 / len(self.linear_bounds)) for i in range(len(self.linear_bounds))]
        print(q_list)
        exit()
        return True

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

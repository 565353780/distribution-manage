import numpy as np


class LinearFunction(object):
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray) -> None:
        self.x_points = x_points
        self.y_points = y_points
        return

    def __call__(self, x_values: np.ndarray) -> np.ndarray:
        y_values = np.zeros_like(x_values)

        y_values = np.interp(x_values, self.x_points, self.y_points)

        left_mask = x_values < self.x_points[0]
        y_values[left_mask] = self.y_points[0] - (
            self.x_points[0] - x_values[left_mask]
        )

        right_mask = x_values > self.x_points[-1]
        y_values[right_mask] = self.y_points[-1] + (
            x_values[right_mask] - self.x_points[-1]
        )

        return y_values

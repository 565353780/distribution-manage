import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from distribution_manage.Method.path import createFileFolder


def plotDistribution(
    data: np.ndarray,
    bins: int = 100,
    save_image_file_path: Union[str, None] = None,
    render: bool = True,
) -> bool:
    num_dimensions = data.shape[1]
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.hist(data[:, i], bins=bins, alpha=0.75, color="blue", edgecolor="black")
        ax.set_title(f"Dimension {i + 1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    for j in range(num_dimensions, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if render:
        plt.show()

    if save_image_file_path is not None:
        createFileFolder(save_image_file_path)

        plt.savefig(save_image_file_path, dpi=300)

    plt.close()
    return True

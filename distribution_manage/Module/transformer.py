import os
import torch
import joblib
import numpy as np
from typing import Union

from distribution_manage.Method.transformer import (
    getTransformerFunction,
    toTransformersFile,
    transformData
)
from distribution_manage.Method.render import plotDistribution


class Transformer(object):
    def __init__(
        self,
        file_path: Union[str, None] = None,
    ) -> None:
        self.transform_dict = {}

        if file_path is not None:
            self.loadFile(file_path)
        return

    @staticmethod
    def fit(
        mode: str,
        data: np.ndarray,
        save_file_path: str,
        overwrite: bool = False,
    ) -> bool:
        transformer_func = getTransformerFunction(mode)
        if transformer_func is None:
            print('[ERROR][Transformer::fit]')
            print('\t getTransformerFunction failed!')
            return False

        if not toTransformersFile(transformer_func, data, save_file_path, overwrite):
            print('[ERROR][Transformer::fit]')
            print('\t toTransformersFile failed!')
            return False

        return True

    def loadFile(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print('[ERROR][Transformer::loadFile]')
            print('\t file not exist!')
            print('\t file_path:', file_path)
            return False

        self.transform_dict = joblib.load(file_path)
        return True

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return transformData(self.transform_dict, data, False)

    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return transformData(self.transform_dict, data, True)

    def plotDistribution(
        self,
        data: np.ndarray,
        bins: int = 100,
        save_image_file_path: Union[str, None] = None,
        render: bool = True,
    ) -> bool:
        if not plotDistribution(data, bins, save_image_file_path, render):
            print('[ERROR][Transformer::plotDistribution]')
            print('\t plotDistribution failed!')
            return False

        return True

import os
import torch
import joblib
import numpy as np
from copy import deepcopy
from typing import Union
from multiprocessing import Pool
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    Binarizer,
    KernelCenterer,
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
)

from distribution_manage.Method.path import removeFile


def getUniformTransformer(data: np.ndarray) -> QuantileTransformer:
    transformer = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', subsample=10000)
    transformer.fit(data.astype(np.float64))
    return transformer

def getNormalTransformer(data: np.ndarray) -> QuantileTransformer:
    transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=10000)
    transformer.fit(data.astype(np.float64))
    return transformer

def getPowerTransformer(data: np.ndarray) -> PowerTransformer:
    transformer = PowerTransformer()
    transformer.fit(data.astype(np.float64))
    return transformer

def getRobustScaler(data: np.ndarray) -> RobustScaler:
    transformer = RobustScaler()
    transformer.fit(data.astype(np.float64))
    return transformer

def getBinarizer(data: np.ndarray) -> Binarizer:
    transformer = Binarizer()
    transformer.fit(data.astype(np.float64))
    return transformer

def getKernelCenterer(data: np.ndarray) -> KernelCenterer:
    transformer = KernelCenterer()
    transformer.fit(data.astype(np.float64))
    return transformer

def getMinMaxScaler(data: np.ndarray) -> MinMaxScaler:
    transformer = MinMaxScaler()
    transformer.fit(data.astype(np.float64))
    return transformer

def getMaxAbsScaler(data: np.ndarray) -> MaxAbsScaler:
    transformer = MaxAbsScaler()
    transformer.fit(data.astype(np.float64))
    return transformer

def getStandardScaler(data: np.ndarray) -> StandardScaler:
    transformer = StandardScaler()
    transformer.fit(data.astype(np.float64))
    return transformer

def toTransformersFile(transformer_func, data: np.ndarray, save_file_path: str, overwrite: bool = False) -> bool:
    if os.path.exists(save_file_path):
        if not overwrite:
            return True

        removeFile(save_file_path)

    data_list = [data[:, i].reshape(-1, 1) for i in range(data.shape[1])]
    print('start create transformers...')
    with Pool(data.shape[1]) as pool:
        transformer_list = pool.map(transformer_func, data_list)

    transformer_dict = {}
    for i in range(len(transformer_list)):
        transformer_dict[str(i)] = transformer_list[i]

    joblib.dump(transformer_dict, save_file_path)
    print('finish save transformers')
    return True

def transformDataWithPool(inputs: list) -> np.ndarray:
    transformer, data, is_inverse = inputs
    if is_inverse:
        trans_data_array = transformer.inverse_transform(data.reshape(-1, 1)).reshape(-1, 1)
    else:
        trans_data_array = transformer.transform(data.reshape(-1, 1)).reshape(-1, 1)
    return trans_data_array

def transformData(transformer_dict: dict, data: Union[np.ndarray, torch.Tensor], is_inverse: bool = False) -> Union[np.ndarray, torch.Tensor]:
    key_num = len(list(transformer_dict.keys()))
    if key_num == 0:
        print('[WARN][mash_distribution::transformData]')
        print('\t transformer dict is empty! will return source data!')
        return data

    if isinstance(data, torch.Tensor):
        data_array = data.detach().clone().cpu().numpy()
    else:
        data_array = deepcopy(data)

    if data_array.ndim != 2:
        valid_data_array = data_array.reshape(-1, data_array.shape[-1])
    else:
        valid_data_array = data_array

    double_data_array = valid_data_array.astype(np.float64)

    inputs_list = [[transformer_dict[str(i)], double_data_array[:, i], is_inverse] for i in range(data.shape[1])]
    with Pool(data.shape[1]) as pool:
        trans_data_array_list = pool.map(transformDataWithPool, inputs_list)

    trans_data_array = np.hstack(trans_data_array_list)

    if data_array.ndim != 2:
        valid_trans_data_array = trans_data_array.reshape(*data_array.shape)
    else:
        valid_trans_data_array = trans_data_array

    if isinstance(data, torch.Tensor):
        valid_trans_data = torch.from_numpy(valid_trans_data_array).to(data.device, dtype=data.dtype)
    else:
        valid_trans_data = valid_trans_data_array.astype(valid_data_array.dtype)

    return valid_trans_data

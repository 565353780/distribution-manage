import numpy as np

from distribution_manage.Module.transformer import Transformer


def demo():
    data = np.linspace(0, 1, 100000, dtype=np.float64).reshape(-1, 1).repeat(25, axis=1)
    save_transformer_file_path = "./output/multi_linear_transformers.pkl"
    bins = 100
    save_image_file_path = "./output/multi_linear_trans_data.pdf"
    render = False
    overwrite = True

    Transformer.fit("multi_linear", data, save_transformer_file_path, overwrite)

    transformer = Transformer(save_transformer_file_path)

    trans_data = transformer.transform(data)

    Transformer.plotDistribution(trans_data, bins, save_image_file_path, render)

    trans_back_data = transformer.inverse_transform(trans_data)

    error_max = np.max(np.abs(data - trans_back_data))

    print("erro_max =", error_max)

    return True

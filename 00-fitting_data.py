import os

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def fit_curve_and_sample_points(x_data, y_data, num_points=None, x_new=None):
    """
    拟合曲线并返回采样点

    Args:
        x_data: 原始x坐标数组
        y_data: 原始y坐标数组
        num_points: 需要生成的点数（可选）
        x_new: 直接指定的x坐标数组（可选）

    Returns:
        tuple: (x_new, y_new) - 拟合后的x和y坐标
    """
    # 创建插值函数
    f = interpolate.interp1d(x_data, y_data, kind='cubic')

    if x_new is not None:
        # 如果直接提供了x_new，则使用它
        y_new = f(x_new)
    elif num_points is not None:
        # 如果提供了num_points，则生成均匀分布的点
        x_new = np.linspace(min(x_data), max(x_data), num_points)
        y_new = f(x_new)
    else:
        raise ValueError("必须提供 num_points 或 x_new 中的一个参数")

    return x_new, y_new


def read_and_fit_data(file_path, num_points=256) -> np.ndarray:
    """
    Args:
        file_path: 数据文件路径，第一列为x, 其余列为y0, y1, y2, ...; 多个函数
        num_points: 采样点数

    Returns:
        tuple: (x_new, y0_new, y1_new, y2_new, ...) - 拟合后的256个点
    """
    # 读取数据
    print(file_path)
    data = np.loadtxt(file_path)
    x_original = data[:, 0]
    y_original = data[:, 1:].transpose()  # 每一行为一个函数

    # 拟合并取num_points个点
    x_new = np.linspace(200, 350, num_points)
    out = [x_new]
    for y_o in y_original:
        _, y_new = fit_curve_and_sample_points(x_original, y_o, x_new=x_new)
        out.append(y_new)
    out = np.array(out).transpose()

    return out


# 示例使用
if __name__ == "__main__":
    print(os.getcwd())
    num_points = 256

    file_path_absorb = 'data/spectra/RS-Ab.txt'
    out_path_absorb = file_path_absorb.replace('.txt', f'-fitted-{num_points}.txt')
    data_absorb = read_and_fit_data(file_path_absorb, num_points=num_points)
    np.savetxt(out_path_absorb, data_absorb)

    file_path_CD = 'data/spectra/RS-CD.txt'
    out_path_CD = file_path_CD.replace('.txt', f'-fitted-{num_points}.txt')
    data_CD = read_and_fit_data(file_path_CD, num_points=num_points)
    np.savetxt(out_path_CD, data_CD)

    out_path_S_left = file_path_CD.replace('-CD.txt', f'-fitted-left-{num_points}.txt')
    data_S_left = data_absorb + data_CD / 2
    np.savetxt(out_path_S_left, data_S_left)

    out_path_S_right = file_path_CD.replace('-CD.txt', f'-fitted-right-{num_points}.txt')
    data_S_right = data_absorb - data_CD / 2
    data_S_right[:, 0] = data_S_left[:, 0]
    np.savetxt(out_path_S_right, data_S_right)

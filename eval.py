import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy import constants as cst

from models.spectra2distribution_net import Spectra2DistributionNet
from losses.loss_functions import loss_cos, loss_l1, calculate_r2
from utils.data_loader import SpectraDataLoader
from config import Config as cfg
from utils.generate_data import gibbs_transform, generate_distributions

def evaluate_model(model_path, device, num_samples=100):
    """评估模型性能"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    # 加载数据
    spectra_vectors_left, spectra_vectors_right = SpectraDataLoader.load_spectra_data(cfg.DATA_FOLDER)
    spectra_vectors_left = torch.from_numpy(spectra_vectors_left).float().to(device)
    spectra_vectors_right = torch.from_numpy(spectra_vectors_right).float().to(device)
    spectra_vectors_mean = torch.mean(spectra_vectors_left, dim=0, keepdim=True)
    lfit_RR, lfit_RS = gibbs_transform('data/GvsTemp')

    # 创建模型
    net = Spectra2DistributionNet(input_dim=cfg.INPUT_DIM,
                                  hidden_dim=cfg.HIDDEN_DIM,
                                  output_dim=cfg.OUTPUT_DIM,
                                  num_backbone=cfg.NUM_BACKBONE,
                                  num_distribution=cfg.NUM_DISTRIBUTION,
                                  num_temperature=cfg.NUM_TEMPERATURE,
                                  random_type=cfg.RANDOM_TYPE,
                                  activation=cfg.ACTIVATION
                                  ).to(device)

    # 加载预训练模型
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print(f"模型已从 {model_path} 加载")

    # 设置为评估模式
    net.eval()

    # 生成测试数据
    inp_temperature, four_rand, inp_distribution = generate_distributions(num_samples, lfit_RR, lfit_RS)
    inp_temperature = inp_temperature.float().to(device)
    four_rand = four_rand.float().to(device)
    inp_distribution = inp_distribution.float().to(device)
    lfit_RR, lfit_RS = lfit_RR.float().to(device), lfit_RS.float().to(device)

    # 计算光谱
    inp_spectra_left = torch.mm(inp_distribution, spectra_vectors_left) - spectra_vectors_mean
    inp_spectra_right = torch.mm(inp_distribution, spectra_vectors_right) - spectra_vectors_mean
    inp_spectra = torch.cat([inp_spectra_left, inp_spectra_right], dim=-1)

    # 添加噪声
    inp_spectra = inp_spectra * (1 + cfg.MEASURE_NOISE_RELATIVE * (torch.rand_like(inp_spectra) - 0.5))
    # 归一化每行到 [-1, 1] 区间
    inp_spectra_min = inp_spectra.min(dim=1, keepdim=True)[0]  # 每行最小值
    inp_spectra_max = inp_spectra.max(dim=1, keepdim=True)[0]  # 每行最大值
    inp_spectra = 2 * (inp_spectra - inp_spectra_min) / (inp_spectra_max - inp_spectra_min) - 1

    with torch.no_grad():
        # 前向传播
        out_four_rand, out_temperature = net(inp_spectra)

        # -------------
        inp_temperature_K = inp_temperature + 273.15  # 换算成绝对温度
        kBT = inp_temperature_K * (cst.k * cst.N_A / cst.calorie * 1e-3)
        # 不同温度下Gibbs自由能
        G_RR = lfit_RR[:, 0].unsqueeze(0) * inp_temperature_K + lfit_RR[:, 1].unsqueeze(0)
        G_SR = lfit_RS[:, 0].unsqueeze(0) * inp_temperature_K + lfit_RS[:, 1].unsqueeze(0)
        # 每一构型下，不同构象间的相对Gibbs自由能
        G_RR = G_RR - torch.min(G_RR, dim=1, keepdim=True)[0]
        G_SR = G_SR - torch.min(G_SR, dim=1, keepdim=True)[0]
        # 构型内格构象相对占比
        dst_RR = torch.softmax(-G_RR / kBT, dim=1)
        dst_SR = torch.softmax(-G_SR / kBT, dim=1)
        # 每个样本分布
        rr_ = dst_RR * out_four_rand[:, 0:1]  # [num_samples, len(dst_RR)]
        ss_ = dst_RR * out_four_rand[:, 1:2]
        sr_ = dst_SR * out_four_rand[:, 2:3]  # [num_samples, len(dst_SR)]
        rs_ = dst_SR * out_four_rand[:, 3:4]
        out_distribution = torch.cat([rr_, ss_, sr_, rs_], dim=1)
        # --------------

        # 转换到CPU进行评估
        inp_temperature = inp_temperature.cpu().numpy()
        out_temperature = out_temperature.cpu().numpy()
        four_rand = four_rand.cpu().numpy()
        out_four_rand = out_four_rand.cpu().numpy()
        inp_distribution = inp_distribution.cpu().numpy()
        out_distribution = out_distribution.cpu().numpy()

        # 计算评估指标
        # 温度评估
        temp_mse_root = root_mean_squared_error(inp_temperature, out_temperature)
        temp_r2 = r2_score(inp_temperature, out_temperature)

        # 构型评估
        config_mse_root = root_mean_squared_error(four_rand, out_four_rand)
        config_r2 = r2_score(four_rand, out_four_rand)

        # 分布评估
        dist_mse_root = root_mean_squared_error(inp_distribution, out_distribution)
        dist_r2_list = [calculate_r2(inp_distribution[i], out_distribution[i]) for i in range(len(inp_distribution))]
        dist_r2_mean = np.mean(dist_r2_list)
        dist_r2_std = np.std(dist_r2_list)

        # 损失函数评估
        loss_config = loss_cos(torch.from_numpy(four_rand), torch.from_numpy(out_four_rand)).item() + \
                          loss_l1(torch.from_numpy(four_rand), torch.from_numpy(out_four_rand)).item()
        loss_temp = loss_l1(torch.from_numpy(out_temperature), torch.from_numpy(inp_temperature)).item()

        # 打印评估结果
        print("-" * 30 + " 模型评估结果 " + "-" * 48)
        print(f"温度预测:")
        print(f"  root-MSE: {temp_mse_root:.6f}")
        print(f"        R²: {temp_r2:.6f}")
        print(f"构型预测:")
        print(f"  root-MSE: {config_mse_root:.6f}")
        print(f"        R²: {config_r2:.6f}")
        print(f"分布预测:")
        print(f"  root-MSE: {dist_mse_root:.6f}")
        print(f"  R² (平均): {dist_r2_mean:.6f} ± {dist_r2_std:.6f}")
        print(f"损失函数:")
        print(f"  构型损失: {loss_config:.6f}")
        print(f"  温度损失: {loss_temp:.6f}")

        return {
            'temperature': {'mse': temp_mse_root, 'r2': temp_r2},
            'configuration': {'mse': config_mse_root, 'r2': config_r2},
            'distribution': {'mse': dist_mse_root, 'r2_mean': dist_r2_mean, 'r2_std': dist_r2_std},
            'losses': {'config': loss_config, 'temp': loss_temp}
        }


def main(model_path, num_samples=10000):
    """主评估函数"""
    # 检查MPS可用性并设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # 模型路径
    model_folders = [os.path.join(model_path, mf) for mf in os.listdir(model_path) if mf.startswith("noise")]
    model_folders.sort()
    acc = []
    for mf in model_folders:
        cfg.MEASURE_NOISE_RELATIVE = float(mf.split('_')[-1])/1000
        print('cfg.MEASURE_NOISE_RELATIVE:', cfg.MEASURE_NOISE_RELATIVE)
        best_path = [f for f in os.listdir(mf) if f.startswith("Spectra2Distribution_model_best")]
        best_path = f'{np.min([float(f.split("_")[-1][:-4]) for f in best_path]):.8f}'
        best_path = f'{mf}/Spectra2Distribution_model_best_{best_path}.pth'
        print(f"{mf}: {best_path}")
        # 运行评估
        results = evaluate_model(best_path, device, num_samples)
        temp_mse_root = results['temperature']['mse']
        temp_r2 = results['temperature']['r2']
        config_mse_root = results['configuration']['mse']
        config_r2 = results['configuration']['r2']
        distribution_mse_root = results['distribution']['mse']
        distribution_r2 = results['distribution']['r2_mean']
        acc.append([cfg.MEASURE_NOISE_RELATIVE,
                    config_mse_root, config_r2,
                    temp_mse_root, temp_r2,
                    distribution_mse_root, distribution_r2])

        if results:
            print(f"{'='*30} {mf} '评估完成' {'='*30}\n")

    acc = np.array(acc)
    np.savetxt(os.path.join(model_path, 'acc.txt'), acc)

if __name__ == '__main__':
    model_path = 'noise_1775'
    main(model_path=model_path, num_samples=100000)

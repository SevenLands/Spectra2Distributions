import torch
import torch.nn.functional as F
import numpy as np

def loss_cos(pre_distribution, tgt_distribution):
    """
    Args:
        pre_distribution: 预测分布集合，形状为 (batch_size, output_dim), 每一行代表一个预测分布
        tgt_distribution: 目标分布集合，形状为 (batch_size, output_dim), 每一行代表一个目标分布
    Returns:
        预测分布与目标分布之间的余弦相似度损失，关注大几率分布间的关系
    """
    # # 余弦相似度损失
    amp_pre, amp_tgt = torch.norm(pre_distribution, dim=1), torch.norm(tgt_distribution, dim=1)
    cos_sim_matrix = torch.sum(pre_distribution * tgt_distribution, dim=1)/(amp_pre * amp_tgt)
    loss_cos = torch.mean(torch.abs(1 - cos_sim_matrix))
    return loss_cos

def loss_mse(pre_distribution, tgt_distribution):
    return F.mse_loss(pre_distribution, tgt_distribution)

def loss_l1(pre_distribution, tgt_distribution):
    return F.l1_loss(pre_distribution, tgt_distribution)

def calculate_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("两条曲线的数值点数量必须相同！")
    # 计算真实值的均值
    y_true_mean = np.mean(y_true)
    # 计算残差平方和（SS_res）和总平方和（SS_tot）
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    # 处理边界情况：若y_true所有值相同（SS_tot=0），R²无意义
    if ss_tot == 0:
        print("警告：参考曲线（y_true）所有值相同，无法计算有效R²！")
        return np.nan
    # 计算R²
    r2 = 1 - ss_res / ss_tot # type: ignore
    return r2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomLinear(nn.Module):
    """
    支持行随机、列随机或行列随机的可训练线性变换层
    """

    def __init__(self, in_features, out_features, random_type='row', bias=False):
        super(RandomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.random_type = random_type

        # 初始化权重矩阵
        if random_type == 'eye':
            if in_features != out_features:
                raise ValueError("For 'eye' random type, in_features must equal out_features.")
            # 将单位矩阵创建在与模块相同的设备上
            weight = torch.eye(out_features)
            self.register_buffer('weight', weight)  # 使用register_buffer而不是直接赋值
        else:
            weight = torch.randn(out_features, in_features)

            # 根据随机类型初始化权重
            if random_type == 'row':
                weight = F.softmax(weight, dim=1)
            elif random_type == 'col':
                weight = F.softmax(weight, dim=0)
            elif random_type == 'both':
                if in_features != out_features:
                    raise ValueError("For 'both' random type, in_features must equal out_features.")
                weight = self._make_doubly_stochastic(weight)
            else:
                raise ValueError(f"Invalid random_type: expected 'row', 'col', 'both', or 'eye'; got {random_type}.")

            # 设置为可训练参数
            self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        if self.random_type == 'eye':
            constrained_weight = self.weight
        elif self.random_type == 'row':
            constrained_weight = F.softmax(self.weight, dim=1)
        elif self.random_type == 'col':
            constrained_weight = F.softmax(self.weight, dim=0)
        elif self.random_type == 'both':
            constrained_weight = self._make_doubly_stochastic(self.weight)
        else:
            constrained_weight = self.weight

        return F.linear(input, constrained_weight, self.bias)

    def _make_doubly_stochastic(self, weight):
        """使用Sinkhorn算法将矩阵转换为双随机矩阵，增加数值稳定性"""
        # 使用数值稳定的softmax方法，避免exp导致的溢出
        weight_stable = weight - weight.max(dim=1, keepdim=True)[0]
        K = torch.exp(weight_stable)

        # 添加小的epsilon值防止除零
        eps = 1e-12
        max_iter = 100
        tolerance = 1e-6

        for i in range(max_iter):
            # 行归一化
            row_sum = K.sum(dim=1, keepdim=True)
            K = K / (row_sum + eps)

            # 列归一化
            col_sum = K.sum(dim=0, keepdim=True)
            K = K / (col_sum + eps)

            # 检查收敛（可选，提高效率）
            if i > 10:  # 前几次迭代不检查，确保足够迭代
                row_sums = K.sum(dim=1)
                col_sums = K.sum(dim=0)
                if (torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance) and
                        torch.allclose(col_sums, torch.ones_like(col_sums), atol=tolerance)):
                    break
        return K

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'random_type={self.random_type}, bias={self.bias is not None}'


class ResidualBlock(nn.Module):
    """
    支持不同输入输出维度的残差块，使用RandomLinear实现残差连接
    """

    def __init__(self, in_features, out_features,
                 activation=nn.ReLU(), random_type='row', bias=True):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            random_type: RandomLinear的随机类型 ('row', 'col', 'both', 'eye')
            bias: 是否使用偏置
            activation: 激活函数
        """
        super(ResidualBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        # 将hidden_features设置为输入维度的一半，最小为1
        self.hidden_features = max(1, in_features // 2)  # 2*in_features  # max(1, in_features // 2)

        # 主路径：输入 -> 隐藏层 -> 输出层
        self.linear1 = nn.Linear(in_features, self.hidden_features, bias=bias)
        self.linear2 = nn.Linear(self.hidden_features, out_features, bias=bias)
        self.activation = activation

        # 残差连接：使用RandomLinear处理维度匹配
        self.shortcut = RandomLinear(in_features, out_features, random_type=random_type)

    def forward(self, x):
        """
        残差块前向传播
        """
        # 主路径
        out = self.activation(self.linear1(x))
        out = self.linear2(out)

        # 残差连接
        shortcut_output = self.shortcut(x)
        out = out + shortcut_output

        return out

    def extra_repr(self):
        return f'in_features={self.in_features}, hidden_features={self.hidden_features}, ' \
               f'out_features={self.out_features}'


class Spectra2DistributionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_backbone=6, num_distribution=1, num_temperature=1,
                 random_type='row', activation='Tanh'):
        super().__init__()

        # 根据字符串选择激活函数
        activation_map = {
            'ReLU': nn.ReLU,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid,
            'GELU': nn.GELU,
        }
        activation_class = activation_map.get(activation, nn.Tanh)

        # 主网络：使用多层ResidualBlock学习传统映射变换
        self.layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_backbone):
            self.layers.append(ResidualBlock(hidden_dim, hidden_dim,
                                             activation=activation_class(), random_type=random_type))
        self.layers = nn.Sequential(*self.layers)

        # 分布：使用一个ResidualBlock和一个线性层学习分布
        self.distribution = []
        for _ in range(num_distribution):
            self.distribution.append(ResidualBlock(hidden_dim, hidden_dim,
                                                   activation=activation_class(), random_type=random_type))
        self.distribution.append(nn.Linear(hidden_dim, output_dim))
        self.distribution = nn.Sequential(*self.distribution)

    def forward(self, x):
        # 输出向量集合，形状为(batch_size, output_dim)
        out = self.layers(x)
        distribution = F.softmax(self.distribution(out), dim=1)
        return distribution

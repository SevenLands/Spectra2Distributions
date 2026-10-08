import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import constants as cst

from tqdm import tqdm

from models.spectra2distribution_net import Spectra2DistributionNet
from losses.loss_functions import loss_cos, loss_l1, calculate_r2
from utils.data_loader import SpectraDataLoader
from utils.lr_lambda import create_lr_lambda
from utils.generate_data import gibbs_transform, generate_distributions
from config import Config as cfg


def main(model_path=None):
    """主训练函数"""
    # 检查MPS可用性并设置设备
    device = cfg.get_device()

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
    print(net)
    #optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-2, betas=(0.9, 0.95))

    # 加载预训练模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=cfg.get_device())
        net.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print("No model found. Training from scratch.")

    num_epochs = cfg.NUM_EPOCHS

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=create_lr_lambda(optimizer,
                                   cos_epochs=num_epochs//5*3,
                                   warmup_epochs=num_epochs//5,
                                   min_lr=cfg.MIN_LR)
    )

    measure_noise_relative = cfg.MEASURE_NOISE_RELATIVE
    out_folder = f'noise_{int(1000*measure_noise_relative):04d}'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    batch_size = cfg.BATCH_SIZE
    loss_record = []
    best_loss = 1e10
    best_net = None
    pbar = tqdm(range(num_epochs + 1))
    for ipar in pbar:  # num_epochs
        inp_temperature, four_rand, inp_distribution = generate_distributions(batch_size, lfit_RR, lfit_RS)
        inp_temperature = inp_temperature.float().to(device)
        four_rand = four_rand.float().to(device)
        inp_distribution = inp_distribution.float().to(device)
        inp_spectra_left = torch.mm(inp_distribution, spectra_vectors_left) - spectra_vectors_mean
        inp_spectra_right = torch.mm(inp_distribution, spectra_vectors_right) - spectra_vectors_mean
        inp_spectra = torch.cat([inp_spectra_left, inp_spectra_right], dim=-1)
        # 添加噪声
        inp_spectra = inp_spectra * (1 + measure_noise_relative * (torch.rand_like(inp_spectra) - 0.5))
        # 归一化每行到 [-1, 1] 区间
        inp_spectra_min = inp_spectra.min(dim=1, keepdim=True)[0]  # 每行最小值
        inp_spectra_max = inp_spectra.max(dim=1, keepdim=True)[0]  # 每行最大值
        inp_spectra = 2 * (inp_spectra - inp_spectra_min) / (inp_spectra_max - inp_spectra_min) - 1

        #for ii in range(2):
        #    plt.plot(inp_spectra[ii].cpu().numpy()[:len(inp_spectra[ii])//2])
        #    plt.plot(inp_spectra[ii].cpu().numpy()[len(inp_spectra[ii])//2:])
        #plt.show()

        optimizer.zero_grad()
        # 前向传播
        out_four_rand, out_temperature = net(inp_spectra)
        loss_conformation = loss_l1(out_four_rand, four_rand) + loss_cos(out_four_rand, four_rand)
        # 温度损失: 1个温度
        loss_temperature = loss_l1(out_temperature, inp_temperature)
        # 总损失
        loss = loss_conformation +  loss_temperature
        loss_record.append([ipar, loss.item(), loss_conformation.item(), loss_temperature.item()])
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"ipar: {ipar}, loss: {loss.item():10.6f}, "
            f"loss_conformation: {loss_conformation.item():10.6f}, "
            f"loss_temperature: {loss_temperature.item():10.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_net = net.state_dict()

        if ipar % cfg.SAVE_INTERVAL == 0:  # 每SAVE_INTERVAL个epoch保存一次
            np.savetxt(out_folder + '/Spectra2Distribution_loss_record.txt', np.array(loss_record))
            torch.save(best_net, out_folder + f'/Spectra2Distribution_model_best_{best_loss:.8f}.pth')
            torch.save(net.state_dict(), out_folder + f'/Spectra2Distribution_model_epoch_{ipar}.pth')
    np.savetxt(out_folder + '/Spectra2Distribution_loss_record.txt', np.array(loss_record))
    torch.save(best_net, out_folder + f'/Spectra2Distribution_model_best_{best_loss:.8f}.pth')

    return out_folder + f'/Spectra2Distribution_model_best_{best_loss:.8f}.pth'

if __name__ == '__main__':
    model_path = 'Spectra2Distribution_model_best_0.13057442-.pth'
    #main(model_path, evaluation=True)

    #'''
    for noise in range(0, 21):
        cfg.MEASURE_NOISE_RELATIVE = noise / 200
        print('-'*50)
        print(f"noise1: {cfg.MEASURE_NOISE_RELATIVE}")
        model_path = main(model_path)
    #'''

"""配置文件"""
import torch

class Config:
    # 模型配置
    INPUT_DIM = 512 + 1
    HIDDEN_DIM = 512
    OUTPUT_DIM = 4
    NUM_BACKBONE = 17
    NUM_DISTRIBUTION = 7
    NUM_TEMPERATURE = 5
    RANDOM_TYPE = 'eye'
    ACTIVATION = 'Tanh'   # 'ReLU'  'Tanh'  'Sigmoid'  'GELU'

    # 数据配置
    DATA_FOLDER = 'data/spectra'
    MEASURE_NOISE_RELATIVE = 0
    MEASURE_NOISE_ABSOLUTE = MEASURE_NOISE_RELATIVE

    # 训练配置
    NUM_EPOCHS = 50000
    LEARNING_RATE = 1e-5
    MIN_LR = 1e-8
    BATCH_SIZE = 500
    SAVE_INTERVAL = NUM_EPOCHS//5

    # 设备配置
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            print("Using MPS device")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA device")
            return torch.device("cuda")
        else:
            print("Using CPU device")
            return torch.device("cpu")

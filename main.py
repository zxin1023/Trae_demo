import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.experiment_manager import ExperimentManager
from configs.default_config import Config
from models.custom_model import CustomModel
from utils.custom_dataset import CustomDataset
from utils.custom_learner import CustomLearner

def prepare_data(config):
    """准备数据加载器"""
    # 加载数据集
    dataset = CustomDataset(config.DATA_DIRC  )
    
    # 划分数据集
    total_size = len(dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    valid_size = int(config.VALID_RATIO * total_size)
    test_size = total_size - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


def main():
    # 加载配置
    config = Config()
    
    # 创建实验管理器
    exp_manager = ExperimentManager(config)
    exp_manager.setup()
    
    # 数据集加载
    train_loader, valid_loader, test_loader = prepare_data(config)

    # 模型创建
    model = CustomModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    
    learner = CustomLearner(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
        exp_manager=exp_manager
    )
    
    # 模型训练
    learner.train(train_loader, valid_loader, test_loader)
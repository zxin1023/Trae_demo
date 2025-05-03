import torch


class Config:
    # 基础配置
    OUTPUT_DIR = 'experiments'
    RESUME = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据集配置
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 训练配置
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    LOG_INTERVAL = 10
    # 学习率调度配置
    STEP_SIZE = 10
    GAMMA = 0.1
    
    # 早停配置
    EARLY_STPPPING_PATIENCE = 10
    
    # 测试配置
    TEST_WITH_BEST_MODEL = True       # 是否使用最佳模型进行测试
    # TEST_WITH_LAST_MODEL = True       # 是否使用最后一个模型进行测试
    # TEST_WITH_AVERAGE_MODEL = True    # 是否使用平均模型进行测试
    SAVE_PREDICTIONS = True           # 是否保存预测结果
    TEST_DURING_TRAINING = False       # 是否在训练过程中进行测试
    TEST_FREQUENCY = 5                # 测试频率

    
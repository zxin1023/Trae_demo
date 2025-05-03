from .. import default_config

class Config(default_config):
    def __init__(self) -> None:
        super(Config).__init__()
        # 数据集配置
        TRAIN_RATIO = 0.8
        VALID_RATIO = 0.1
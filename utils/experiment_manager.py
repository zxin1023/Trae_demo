import os
import json
import ymal
import time
import shutil
from pathlib import Path


class ExperimentManager:
    """实验管理器类
    
    用于管理深度学习实验的完整生命周期，包括:
    - 创建和维护实验目录结构
    - 保存实验配置和检查点
    - 追踪实验元数据
    - 管理模型检查点
    
    Attributes:
        config: 实验配置对象
        exp_id: 实验唯一标识符
        exp_dir: 实验根目录路径
    """
    def __init__(self, config) -> None:
        self.config = config
        self.exp_id = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(config.EXP_DIR) / f"exp_{self.exp_id}"
        self.set_experiment_dir()
    
    def set_experiment_dir(self):
        # 创建实验目录结构
        # 创建主目录
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "configs").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        (self.exp_dir / "visualization").mkdir(exist_ok=True)
        
        # 保存配置快照
        self.save_config()
        
        # 初始化元数据
        self.init_metadata()
    
    def save_config(self):
        """保存完整配置"""
        config_path = self.exp_dir / "configs/config_dump.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            ymal.dump(self.config.__dict__, f)
            
    def init_metadata(self):
        """初始化实验元数据"""
        metadata = {
            'exp_id': self.exp_id,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'running',
            'current_epoch': 0,
            'best_metric': None,
            'total_training_time': 0
        }
        self.save_metadata(metadata)
    
    def save_metadata(self, metadata):
        """保存元数据"""
        with open(self.exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def get_checkpoint_path(self, epoch):
        """获取检查点路径"""
        return self.exp_dir / f"checkpoints/epoch_{epoch}.pth"
    
    def update_latest_link(self, checkpoint_path):
        """更新最新检查点的软链接"""
        latest_link = self.exp_dir / "checkpoints/latest.pth"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path)
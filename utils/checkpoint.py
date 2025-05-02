import time
# from pathlib import Path

import torch


class CheckpointManager:
    """管理模型检查点的保存和加载
    
    该类负责处理模型检查点的保存和加载操作，包括:
    - 保存常规检查点
    - 保存最佳模型
    - 更新最新检查点链接
    - 加载指定检查点
    
    Attributes:
        exp_manager: 实验管理器实例，用于管理实验相关的路径和配置
        best_metric: 记录最佳评估指标的值
    """
    def __init__(self, exp_manager) -> None:
        self.exp_manager = exp_manager
        self.best_metric = float('inf')

    def save_checkpoint(self, state, epoch, is_best=False):
        """保存模型检查点"""
        # 添加额外信息
        state.update({
            'timestamp': time.strftime("%Y%m%d-%H%M%S"),
            'exp_id': self.exp_manager.exp_id,
            'config': self.exp_manager.config.__dict__
        })

        # 保存常规检查点
        checkpoint_path = self.exp_manager.get_checkpoint_path(epoch)
        torch.save(state, checkpoint_path)

        # 更新最新检查点链接
        self.exp_manager.update_latest_link(checkpoint_path)
        
        # 如果是最佳模型，单独保存
        if is_best:
            best_path = self.exp_manager.exp_dir / "checkpoint/best_model.pth"
            torch.save(state, best_path)
        
        # 更新元数据
        metadata = {
            'current_epoch': epoch, 
            'best_metric': self.best_metric if is_best else state.get('best_metric'),
            'last_checkpoint': str(checkpoint_path),
            'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.exp_manager.save_metadata(metadata)
        
    def load_checkpoint(self, path=None):
        """加载检查点"""
        if path is None:
            # 默认加载最新检查点
            path = self.exp_manager.exp_dir / "checkpoints/latest.pth"
        
        if not path.exists():
            return None
        
        return torch.load(path)
        
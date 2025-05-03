from copy import deepcopy
import logging
# from tqdm import tqdm
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import metrics
from utils.metrics import AverageMeter
from utils.checkpoint import CheckpointManager


class BaseLearner:
    """基础学习器类
    
    这是一个基础的机器学习训练器类，提供了训练、验证、测试等基本功能。
    包含了模型训练的完整生命周期管理，包括:
    - 训练循环控制
    - 模型验证
    - 模型测试
    - 检查点保存和恢复
    - 早停机制
    - 指标记录和可视化
    - 日志记录
    """
    def __init__(self, model, optimizer, criterion, scheduler, config, exp_manager, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.exp_manager = exp_manager
        self.config = config
        self.checkpoint_manager = CheckpointManager(self.exp_manager.exp_dir)
        
        self.writer = SummaryWriter(exp_manager.exp_dir / "logs/tensorboard")
        self.best_val_metrics = float('inf')
        self.setup_logging()
        
        # 保存最佳模型的状态字典副本
        self.best_model_state_dict = None
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.exp_manager.exp_dir / "logs/train.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
        logging.info("Logging setup complete.")
        
    def log_metrics(self, mode, metrics, epoch):
        """记录指标到日志和TensorBoard."""
        log_str = f"Epoch {epoch} {mode.capitalize()} Metrics: "
        metric_strs = []
        for name, value in metrics.items():
            # 记录到Tensorboard
            self.writer.add_scalar(f'{mode}/{name}', value, epoch)
            metric_strs.append(f'{name}: {value:.4f}')
        log_str += ", ".join(metric_strs)
        logging.info(log_str)
        
    def log_progress(self, mode, epoch, batch_idx, total_batches, metrics):
        if isinstance(metrics, str):   # 处理简单消息, 例如“Training finished!”
            logging.info(metrics)
            return 
        log_str = f"{mode.capitalize()} Epoch: {epoch} [{batch_idx} {total_batches}]"
        metric_strs = []
        # metrics 可能是一个 AverageMeter 对象或一个字典
        avg_metrics = metrics.average() if hasattr(metrics, 'average') else metrics
        for name, value in avg_metrics.items():
            metric_strs.append(f"{name}: {value:.4f}")
        log_str += " " + ", ".join(metric_strs)
        logging.info(log_str)
        
    def save_visualization(self, epoch, images, prefix=""):
        """保存可视化结果
        
        将模型生成的图像保存到指定目录。
        
        Args:
            epoch (int): 当前训练的轮次
            images (list): 需要保存的图像列表
            prefix (str, optional): 图像文件名的前缀. Defaults to "".
            
        Returns:
            None
        """
        vis_dir = self.exp_manager.exp_dir / f"visualizations/epoch_{epoch}"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            save_path = vis_dir / f"{prefix}_{idx}.png"
            img.save(save_path)
            
            
    def calculate_accuracy(self, outputs, targets):
        """计算模型预测准确率
        
        Args:
            outputs (torch.Tensor): 模型的输出预测结果，形状为 (batch_size, num_classes)
            targets (torch.Tensor): 真实标签，形状为 (batch_size,)
            
        Returns:
            float: 预测准确率，范围在 [0, 1] 之间
        """
        with torch.no_grad():
            # 获取预测概率最高的类别索引
            _, predicted = torch.max(outputs.data, 1)
            # 计算总样本数
            total = targets.size(0)
            # 计算预测正确的样本数
            correct = (predicted == targets).sum().item()
            # 返回准确率
            return correct / total

    def train_one_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        metrics = AverageMeter()     # 记录指标
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.EPOCHS}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            # 更新指标
            train_loss = loss.item()
            train_acc = self.calculate_accuracy(output, targets)
            metrics.update("train_loss", train_loss)
            metrics.update("train_acc", train_acc)
            pbar.set_postfix(metrics.average())
            
            # 记录训练过程
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.log_progress("Train", epoch, batch_idx, len(train_loader), metrics)
                
            # 记录到tensorboard
            self.writer.add_scalar("train_loss", train_loss, epoch * len(train_loader) + batch_idx)
            self.writer.add_scalar("train_acc", train_acc, epoch * len(train_loader) + batch_idx)

        return metrics.average()
    
    def train(self, train_loader, valid_loader, test_loader=None):
        """完整训练流程"""
        start_epoch = 0
        
        # 恢复训练
        if self.config.RESUME:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                self.restore_training_state(checkpoint)
                start_epoch = checkpoint['epoch'] + 1
                logging.info("Resuming training from epoch %d", start_epoch)
            else:
                logging.info("No checkpoint found. Starting training from scratch.")
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            # 训练
            train_metrics = self.train_one_epoch(train_loader, epoch)
            self.log_metrics('train', train_metrics, epoch)
        
            # 验证
            val_metrics = self.validate(valid_loader, epoch)
            self.log_metrics('val', val_metrics, epoch)

            # 检查是否是最佳模型
            is_best = val_metrics['val_loss'] < self.best_val_metrics['val_loss']
            if is_best:
                self.best_val_metrics = val_metrics['val_loss']

                # 如果有测试集，使用最佳模型进行测试
                if test_loader and self.config.TEST_DURING_TRAINING:
                    test_metrics = self.test(test_loader, use_best_model=self.config.TEST_WITH_BEST_MODEL)
                    self.log_progress('test', epoch, 0, 1, test_metrics['metrics'])
            
            # 保存检查点
            self.save_training_state(epoch, train_metrics, val_metrics, is_best)

            # 学习率调整
            self.scheduler.step()

            # 早停
            if self.early_stopping(val_metrics['val_loss']):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # 记录到tensorboard
            # for name, value in {**train_metrics, **val_metrics}.items():
            #     self.writer.add_scalar(name, value, epoch)
        
        # 训练结束后的最终测试
        if test_loader and not self.config.TEST_WITH_BEST_MODEL:
            logging.info("Performing final test...")
            # 加载最佳模型
            # self.checkpoint_manager.load_best_model()
            test_results = self.test(test_loader)
            self.log_metrics('final_test', test_results['metrics'], self.config.EPOCHS)


        # 训练结束后，关闭tensorboard   
        self.writer.close()
        logging.info("Training finished!")

    def test(self, test_loader, use_best_model=True):
        """测试模型"""
        # 保存当前模型状态
        current_model_state = deepcopy(self.model.state_dict())
        
        # 如果使用最佳模型，加载最佳模型状态
        if use_best_model:
            # self.checkpoint_manager.load_best_model()
            if self.best_model_state is None:
                logging.warning("No best model state found, using current model state")
            else:
                self.model.load_state_dict(self.best_model_state)
         
        # 测试       
        self.model.eval()
        metrics = AverageMeter()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 更新指标
                test_loss = loss.item()
                test_acc = self.calculate_accuracy(outputs, targets)
                metrics.update("test_loss", test_loss)
                metrics.update("test_acc", test_acc)

                # 记录预测结果
                predictions.extend(outputs.cpu().numpy())
                targets.extend(targets.cpu().numpy())

        # 保存测试结果
        test_results = {
            'metrics': metrics.average(),
            'predictions': predictions,
            'targets': targets,
            'model_type': 'best' if use_best_model else 'current'
        }
        self.save_test_results(test_results)
        
        # 如果使用了最佳模型，恢复到当前模型状态
        if use_best_model:
            self.model.load_state_dict(current_model_state)
        
        return test_results

    def save_test_results(self, test_results):
        """保存测试结果"""
        results_dir = self.exp_manager.exp_dir / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"test_results_{test_results['model_type']}.pth"
        torch.save(test_results, results_file)
        logging.info("Test results saved to %s", results_file)      
    
    def validate(self, valid_loader, epoch):
        """验证模型"""
        self.model.eval()
        metrics = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 更新指标
                val_loss = loss.item()
                val_acc = self.calculate_accuracy(outputs, targets)
                metrics.update("val_loss", val_loss)
                metrics.update("val_acc", val_acc)
        avg_metrics = metrics.average()
        
        # 如果是最佳模型，保存模型状态的深度复制
        if avg_metrics['val_loss'] < self.best_val_metrics['val_loss']:
            self.best_val_metrics = avg_metrics['val_loss']
            self.best_model_state_dict = deepcopy(self.model.state_dict())
        
        # 记录验证过程
        self.log_progress('val', epoch, 0, 1, metrics)
        return avg_metrics

    def save_training_state(self, epoch, train_metrics, val_metrics, is_best):
        """保存训练状态"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_metrics': self.best_val_metrics,
            'config': self.config.__dict__,
            'is_best': is_best        
        }
        self.checkpoint_manager.save_checkpoint(state, epoch, is_best)

    def early_stopping(self, val_loss):
        """早停检查"""
        if not hasattr(self, 'early_stopping_counter'):
            self.early_stopping_counter = 0
            self.best_loss = float('inf')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE

    def restore_training_state(self, checkpoint):
        """恢复训练状态"""
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.best_model_state_dict = checkpoint['model_state_dict']
        
        


    
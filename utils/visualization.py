import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_metrics(metrics_history, title='Training Metrics', save_path=None):
    """
    绘制训练过程中的指标变化曲线。

    Args:
        metrics_history (dict): 包含指标历史记录的字典。
                                键是指标名称 (e.g., 'train_loss', 'val_acc')，
                                值是包含每个 epoch 指标值的列表。
        title (str): 图表标题。
        save_path (str or Path, optional): 图表保存路径。如果提供，图表将保存到文件。
    """
    num_metrics = len(metrics_history)
    if num_metrics == 0:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, num_metrics * 4), sharex=True)
    if num_metrics == 1:
        axes = [axes] # 保证 axes 是可迭代的

    epochs = range(1, len(next(iter(metrics_history.values()))) + 1)

    for ax, (metric_name, values) in zip(axes, metrics_history.items()):
        ax.plot(epochs, values, marker='o', linestyle='-')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True)
        if ax == axes[-1]: # 只在最下面的子图显示 x 轴标签
            ax.set_xlabel('Epoch')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig) # 关闭图形，释放内存

# 可以在训练结束后调用:
# from utils.visualization import plot_metrics
# history = {'train_loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.3], 'train_acc': [0.8, 0.9, 0.95], 'val_acc': [0.75, 0.85, 0.9]}
# plot_metrics(history, save_path=exp_manager.exp_dir / 'visualizations/training_curves.png')

def save_image_grid(images, save_path, grid_size=(4, 4), title=None):
    """
    将一组图像保存为网格图。

    Args:
        images (list or np.ndarray): 图像列表或 NumPy 数组 (N, H, W, C) 或 (N, H, W)。
        save_path (str or Path): 网格图保存路径。
        grid_size (tuple): 网格的行数和列数。
        title (str, optional): 图表标题。
    """
    images = np.asarray(images)
    num_images = images.shape[0]
    rows, cols = grid_size
    
    if num_images > rows * cols:
        print(f"Warning: Displaying only the first {rows * cols} images out of {num_images}.")
        images = images[:rows * cols]
        num_images = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() # 将二维数组展平

    for i in range(num_images):
        img = images[i]
        # 处理灰度图和彩色图
        if img.ndim == 2: # 灰度图 (H, W)
            axes[i].imshow(img, cmap='gray')
        elif img.ndim == 3: # 彩色图 (H, W, C)
             # 检查通道数是否在最后，如果不是，尝试调整
            if img.shape[0] in [1, 3]: # (C, H, W)
                img = np.transpose(img, (1, 2, 0))
            # 归一化到 [0, 1] (如果需要)
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1) # 确保值在有效范围内
            axes[i].imshow(img)
        
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Image grid saved to {save_path}")
    plt.close(fig)

# 可以在需要可视化图像时调用，例如在 BaseLearner 的 validate 或 test 方法中
# from utils.visualization import save_image_grid
# sample_images = [np.random.rand(32, 32, 3) for _ in range(16)] # 假设这是 16 张 32x32 的彩色图像
# save_image_grid(sample_images, exp_manager.exp_dir / 'visualizations/sample_batch.png')
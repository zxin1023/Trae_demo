

from collections import defaultdict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.metrics  = defaultdict(lambda: {'sum': 0, 'count': 0})
        
    def update(self, metric_name, value, n=1):
        self.metrics[metric_name]['sum'] += value * n
        self.metrics[metric_name]['count'] += n
        
    def average(self):
        return {name: metrics['sum'] / metrics['count'] for name, metrics in self.metrics.items()}
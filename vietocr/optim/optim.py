from torch.optim import lr_scheduler


class CosineAnnealingLR(object):
    def __init__(self, epochs, step_each_epoch, warmup_epoch=0, last_epoch=-1, **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step, num_cycles=0.5):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        progress = float(current_step - self.warmup_epoch) / float(max(1, self.epochs - self.warmup_epoch))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

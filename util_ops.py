import os
import shutil
from bisect import bisect_right

import pandas as pd
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def record_info(info, filename, mode):
    if mode == 'train':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} '
            'Data {data_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5}\n'
            'LR {lr}\n'.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                               data_time=info['Data Time'], loss=info['Loss'],
                               top1=info['Prec@1'], top5=info['Prec@5'], lr=info['lr']))
        print(result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Data Time', 'Loss', 'Prec@1', 'Prec@5', 'lr']

    if mode == 'test':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5} \n'.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                                      loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Epoch Time', 'Loss', 'Prec@1', 'Prec@5']

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)


class WarmUpMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, warm_up_epochs=5):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warm_up_epochs = warm_up_epochs
        super(WarmUpMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            return [base_lr / self.warm_up_epochs * (self.last_epoch + 1)  # self.last_epoch is begin from 0
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    epoch_name = 'epoch_%d_' % (epoch)
    filename = '_'.join(filename)
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((epoch_name, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

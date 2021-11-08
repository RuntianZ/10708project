'''
https://github.com/locuslab/deq/blob/master/MDEQ-Vision/lib/utils/utils.py
'''

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, train_step=-1, **kwargs):
    outputs, jac_loss, sradius = self.model(inputs, train_step=train_step, **kwargs)
    loss = self.loss(outputs, labels)
    return loss.unsqueeze(0), jac_loss.unsqueeze(0), outputs, sradius


def get_world_size():
  if not torch.distributed.is_initialized():
    return 1
  return torch.distributed.get_world_size()


def get_rank():
  if not torch.distributed.is_initialized():
    return 0
  return torch.distributed.get_rank()


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.initialized = False
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def initialize(self, val, weight):
    self.val = val
    self.avg = val
    self.sum = val * weight
    self.count = weight
    self.initialized = True

  def update(self, val, weight=1):
    if not self.initialized:
      self.initialize(val, weight)
    else:
      self.add(val, weight)

  def add(self, val, weight):
    self.val = val
    self.sum += val * weight
    self.count += weight
    self.avg = self.sum / self.count

  def value(self):
    return self.val

  def average(self):
    return self.avg


def create_logger(log_path, phase='train'):

  if not os.path.isdir(log_path):
    os.makedirs(log_path)

  time_str = time.strftime('%Y-%m-%d-%H-%M')
  log_file = 'mnist_{}_{}.log'.format(time_str, phase)
  final_log_file = os.path.join(log_path, log_file)
  head = '%(asctime)-15s %(message)s'
  logging.basicConfig(filename=str(final_log_file),
                      format=head)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  console = logging.StreamHandler()
  logging.getLogger('').addHandler(console)

  return logger


def get_optimizer(cfg, model):
  optimizer = None
  if cfg.TRAIN.OPTIMIZER == 'sgd':
    optimizer = optim.SGD(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=cfg.TRAIN.LR,
      momentum=cfg.TRAIN.MOMENTUM,
      weight_decay=cfg.TRAIN.WD,
      nesterov=cfg.TRAIN.NESTEROV
    )
  elif cfg.TRAIN.OPTIMIZER == 'adam':
    optimizer = optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=cfg.TRAIN.LR,
      weight_decay=cfg.TRAIN.WD,
    )
  elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
    optimizer = optim.RMSprop(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=cfg.TRAIN.LR,
      momentum=cfg.TRAIN.MOMENTUM,
      weight_decay=cfg.TRAIN.WD,
      alpha=cfg.TRAIN.RMSPROP_ALPHA,
      centered=cfg.TRAIN.RMSPROP_CENTERED
    )

  return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
  torch.save(states, os.path.join(output_dir, filename))
  if is_best and 'state_dict' in states:
    torch.save(states['state_dict'],
               os.path.join(output_dir, 'model_best.pth.tar'))


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
  """
  Calcute the confusion matrix by given label and pred
  """
  output = pred.cpu().numpy().transpose(0, 2, 3, 1)
  seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
  seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

  ignore_index = seg_gt != ignore
  seg_gt = seg_gt[ignore_index]
  seg_pred = seg_pred[ignore_index]

  index = (seg_gt * num_class + seg_pred).astype('int32')
  label_count = np.bincount(index)
  confusion_matrix = np.zeros((num_class, num_class))

  for i_label in range(num_class):
    for i_pred in range(num_class):
      cur_index = i_label * num_class + i_pred
      if cur_index < len(label_count):
        confusion_matrix[i_label,
                         i_pred] = label_count[cur_index]
  return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
  lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
  optimizer.param_groups[0]['lr'] = lr
  return lr
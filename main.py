import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import MNIST
from torchvision import transforms

from model import MnistGan
from train import train, generate_imgs
from deq.utils import create_logger

import matplotlib.pyplot as plt


class GanLoss(nn.Module):
  def __init__(self):
    super(GanLoss, self).__init__()

  def forward(self, x, y):
    # y - 0 for fake img, 1 for real img
    if torch.isinf(x).sum() > 0:
      print(x)
      raise RuntimeError
    loss = y * F.logsigmoid(-x) + (1 - y) * F.logsigmoid(x)
    loss = loss[~torch.isnan(loss)]
    if len(loss) == 0:
      return None
    return loss.mean()


def recover_imgs(imgs):
  print(imgs)
  imgs[imgs < 0] = 0
  imgs[imgs > 1] = 1
  imgs *= 255
  imgs = imgs.int()
  return imgs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('config_file', type=str)
  parser.add_argument('--generate', default=False, action='store_true')
  args = parser.parse_args()

  config_file = args.config_file
  logger = create_logger('log')
  with open(config_file, 'r') as f:
    s = f.read()
    config = json.loads(s)
    logger.info('Config:\n{}\n'.format(config))

  if args.generate:
    state_dict = torch.load(config['save_path'], map_location=torch.device('cpu'))
    model = MnistGan(**config)
    model = model.to(config['device'])
    model.load_state_dict(state_dict['model'])
    imgs = generate_imgs(model, 1, **config)
    imgs = recover_imgs(imgs).view(28, 28)
    plt.imshow(imgs, cmap='Greys')
    plt.show()


  else:

    transform_train = transforms.Compose([transforms.ToTensor()])
    dataset_train = MNIST(root=config['data_root'], download=True, transform=transform_train)
    loader_train = DataLoader(dataset_train, batch_size=config['batch_imgs'],
                              shuffle=True, num_workers=4, drop_last=True)

    model = MnistGan(**config)
    model = model.to(config['device'])
    optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['wd'],
                    momentum=config['momentum'])
    criterion = GanLoss()
    if config.get('scheduler'):
      milestones = config['scheduler'].split(',')
      milestones = [int(s) for s in milestones]
      scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
      config['scheduler'] = scheduler

    train_steps, num_epochs = train(model, loader_train, criterion, optimizer, **config)

    if config.get('save_path'):
      state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_steps': train_steps,
        'epochs': num_epochs,
      }
      torch.save(state_dict, config['save_path'])

if __name__ == '__main__':
  main()

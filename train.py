import logging
import time
import torch

from deq.utils import AverageMeter
from model import MnistGan

logger = logging.getLogger(__name__)
PRETRAIN_STEPS = 1000

def train_epoch(model, train_loader, criterion, optimizer, **kwargs):
  '''
  Train a GAN DEQ for one epoch.
  
  train_loader: generates (x_noise, x_realimg) (Note: No labels!)
                Labels auto generated (0 for fake img, 1 for real img)

  Returns: train_steps - # training steps up till end of this epoch
  '''
  model.train()
  device = model.device
  train_steps = kwargs.get('train_steps', 0)
  pretrain_steps = kwargs.get('pretrain_steps', PRETRAIN_STEPS)
  scheduler = kwargs.get('scheduler')

  losses = AverageMeter()
  total_samples = 0
  total_correct = 0
  t1 = time.time()

  for _, (x_noise, x_realimg) in enumerate(train_loader):
    x_noise, x_realimg = x_noise.to(device), x_realimg.to(device)
    target = torch.cat(torch.zeros((len(x_noise),)), torch.ones((len(x_realimg),))).long().to(device)
    total_samples += len(x_noise) + len(x_realimg)
    train_steps += 1
    deq_mode = (train_steps > pretrain_steps)

    y, _, _ = model(x_noise, x_realimg, deq_mode=deq_mode, compute_jac_loss=False)
    prediction = (y > 0).long()
    correct = (prediction == target)
    total_correct += correct

    loss = criterion(y, target)
    losses.update(loss.item(), len(x_noise) + len(x_realimg))

    optimizer.zero_grad()
    loss.backward()
    # The generator needs to maximize the loss, so reverse the gradients
    for name, param in model.named_parameters():
      if name.startswith('gen') and param.grad is not None:
        param.grad = -param.grad

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

  t2 = time.time()
  t = t2 - t1
  acc = total_correct / total_samples
  msg = 'Elapsed time:           {time}\n' \
        'Total Training samples: {sample}\n' \
        'Training Accuracy:      {acc}\n' \
        'Average Training Loss:  {loss}\n\n'.format(
          time=t, sample=total_samples, acc=acc, loss=losses.avg
        )
  logger.info(msg)

  return train_steps


def train(model, data_loader, criterion, optimizer,
          num_epochs, num_noise, **kwargs):
  '''
  Arguments:
  data_loader:      generates (x, y) (y is not used in training a GAN)
  num_noise:        number of fake samples each itearation
  sigma:            sigma of the low-dim Gaussian distribution
  scheduler:        lr scheduler
  pretrain_steps:   number of pretraining steps
  '''
  model.train()
  train_steps = 0
  dim_noise = kwargs.get('dim_noise', MnistGan.DIM_NOISE)
  sigma = kwargs.get('sigma', 1.0)

  # Build train_loader
  def train_loader():
    for _, (x_realimg, y) in enumerate(data_loader):
      x_noise = torch.randn((num_noise, dim_noise)) * sigma
      yield x_noise, x_realimg

  for i in range(num_epochs):
    logger.info('==> Epoch {}\n'.format(i + 1))
    train_steps = train_epoch(model, train_loader(), criterion, optimizer,
                              train_steps=train_steps, **kwargs)


def generate_imgs(model, num_imgs, dim_noise=None, 
                  sigma=1.0, deq_mode=True):
  if dim_noise is None:
    dim_noise = MnistGan.DIM_NOISE
  x_noise = torch.randn((num_imgs, dim_noise)) * sigma
  imgs = model.generate(x_noise, deq_mode=deq_mode)
  return imgs
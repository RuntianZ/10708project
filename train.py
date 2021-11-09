import logging
import time
import torch
from torch.nn.utils import clip_grad_norm_

from deq.utils import AverageMeter
from model import MnistGan

logger = logging.getLogger(__name__)
PRETRAIN_STEPS = 1000
DEFAULT_SIGMA = 1.0
NUM_EPOCHS = 100
NUM_NOISE = 128
GEN_FACTOR = 1.0

def train_epoch(model, train_loader, criterion, optimizer, **kwargs):
  '''
  Train a GAN DEQ for one epoch.
  
  train_loader: generates (x_noise, x_realimg) (Note: No labels!)
                Labels auto generated (0 for fake img, 1 for real img)

  Returns: train_steps - # training steps up till end of this epoch
  '''
  model.train()
  device = kwargs.get('device')
  train_steps = kwargs.get('train_steps', 0)
  pretrain_steps = kwargs.get('pretrain_steps', PRETRAIN_STEPS)
  scheduler = kwargs.get('scheduler')
  # Apply a different lr (* gen_factor) to the generator
  gen_factor = kwargs.get('gen_factor', GEN_FACTOR)
  gradient_clip = kwargs.get('gradient_clip')

  losses = AverageMeter()
  total_samples = 0
  total_correct = 0
  t1 = time.time()

  for _, (x_noise, x_realimg) in enumerate(train_loader):
    x_noise, x_realimg = x_noise.to(device), x_realimg.to(device)
    target = torch.cat((torch.zeros((len(x_noise),)), torch.ones((len(x_realimg),)))).long().to(device)
    total_samples += len(x_noise) + len(x_realimg)
    train_steps += 1
    deq_mode = (train_steps > pretrain_steps)

    img_noise = torch.randn_like(x_realimg) * 0.05
    x_realimg += img_noise

    y, _, _ = model(x_noise, x_realimg, deq_mode=deq_mode, compute_jac_loss=False)
    y = y.flatten()
    prediction = (y > 0).long()
    a = (prediction == target)
    correct = a.sum()
    total_correct += correct.item()

    correct_fake = a[:len(x_noise)].sum().item()
    correct_real = a[len(x_noise):].sum().item()
    # print('{}\t{}\t{}'.format(train_steps, correct_fake, correct_real))

    loss = criterion(y, target)
    if loss:
      # print(loss.item())
      losses.update(loss.item(), len(x_noise) + len(x_realimg))

      optimizer.zero_grad()
      loss.backward()

      if gradient_clip:
        clip_grad_norm_(model.parameters(), gradient_clip)

      # The generator needs to maximize the loss, so reverse the gradients
      for name, param in model.named_parameters():
        if name.startswith('gen') and param.grad is not None:
          # print(name)
          param.grad = -param.grad * gen_factor
        # if name.startswith('dis') and param.grad is not None:
        #   param.grad *= 0

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


def train(model, data_loader, criterion, optimizer, **kwargs):
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
  sigma = kwargs.get('sigma', DEFAULT_SIGMA)
  num_epochs = kwargs.get('num_epochs', NUM_EPOCHS)
  num_noise = kwargs.get('num_noise', NUM_NOISE)

  # Build train_loader
  def train_loader():
    for _, (x_realimg, y) in enumerate(data_loader):
      x_noise = torch.randn((num_noise, dim_noise)) * sigma
      yield x_noise, x_realimg

  for i in range(num_epochs):
    logger.info('==> Epoch {}\n'.format(i + 1))
    train_steps = train_epoch(model, train_loader(), criterion, optimizer,
                              train_steps=train_steps, **kwargs)

  return train_steps, num_epochs


def generate_imgs(model, num_imgs, **kwargs):
  '''
  Generate a batch of (num_imgs) fake images.
  '''
  dim_noise = kwargs.get('dim_noise', MnistGan.DIM_NOISE)
  sigma = kwargs.get('sigma', DEFAULT_SIGMA)
  deq_mode = kwargs.get('deq_mode', True)

  if dim_noise is None:
    dim_noise = MnistGan.DIM_NOISE
  x_noise = torch.randn((num_imgs, dim_noise)) * sigma
  imgs = model.generate(x_noise, deq_mode=deq_mode)
  return imgs
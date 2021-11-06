import logging
import time
import torch

from deq.utils import AverageMeter

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

    y, _ = model(x_noise, x_realimg, deq_mode=deq_mode, compute_jac_loss=False)
    prediction = (y > 0).long()
    correct = (prediction == target)
    total_correct += correct

    loss = criterion(y, target)
    losses.update(loss.item(), len(x_noise) + len(x_realimg))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()

  t2 = time.time()
  t = t2 - t1
  acc = total_correct / total_samples
  msg = 'Elapsed time:  {time}\n' \
        'Total samples: {sample}\n' \
        'Accuracy:      {acc}\n' \
        'Average Loss:  {loss}\n\n'.format(
          time=t, sample=total_samples, acc=acc, loss=losses.avg
        )
  logger.info(msg)

  return train_steps
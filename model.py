import numpy as np
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from deq.layer_utils import conv3x3, list2vec, vec2list
from deq.core.jacobian import jac_loss_estimate, power_method
from deq.core.solvers import broyden


class MnistGan(nn.Module):
  '''
  Generator and discriminator for MNIST
  Generator: 100 -> (64, 7, 7) -> (32, 14, 14) -> (1, 28, 28)
  Discriminator: (1, 28, 28) -> (32, 14, 14) -> (64, 7, 7) -> 3136 -> 1
  '''

  # Default configs
  DIM_NOISE = 100
  NUM_GROUPS = 4
  NUM_LAYERS = 5
  BLOCK_GN_AFFINE = True
  F_THRES = 30
  B_THRES = 40
  STOP_MODE = 'abs'

  def __init__(self, **kwargs):
    super(MnistGan, self).__init__()
    self.config = kwargs
    self.device = self.config.get('device')
    dim_noise = self.config.get('dim_noise', self.DIM_NOISE)
    num_groups = self.config.get('num_groups', self.NUM_GROUPS)
    block_gn_affine = self.config.get('block_gn_affine', self.BLOCK_GN_AFFINE)

    # Build generator
    self.gen_block_first = nn.Linear(dim_noise, 7 * 7 * 64)
    self.gen_blocks = nn.ModuleList([
      nn.Sequential(
        conv3x3(64, 64), 
        # nn.GroupNorm(num_groups, 64, affine=block_gn_affine), 
        nn.LeakyReLU(0.2)),
      nn.Sequential(
        conv3x3(32, 32), 
        # nn.GroupNorm(num_groups, 32, affine=block_gn_affine), 
        nn.LeakyReLU(0.2)),
    ])
    self.gen_fuse_blocks = nn.ModuleList([
      nn.ModuleList([nn.Identity(), self._upsample_module(64, 32)]),
      nn.ModuleList([self._downsample_module(32, 64), nn.Identity()]),
    ])  # fuse[i][j] = i -> j
    self.gen_block_last = self._upsample_module(32, 1)
    self.gen_shapes = [
      [-1, 64, 7, 7],
      [-1, 32, 14, 14],
    ]  # Shapes of block outputs: The first dimension is the batch size (to be filled in later)

    # Build discriminator
    self.dis_block_first = self._downsample_module(1, 32)
    self.dis_blocks = nn.ModuleList([
      nn.Sequential(
        conv3x3(32, 32), 
        # nn.GroupNorm(num_groups, 32, affine=block_gn_affine), 
        nn.ReLU()),
      nn.Sequential(
        conv3x3(64, 64),
        #  nn.GroupNorm(num_groups, 64, affine=block_gn_affine), 
         nn.ReLU()),
    ])
    self.dis_fuse_blocks = nn.ModuleList([
      nn.ModuleList([nn.Identity(), self._downsample_module(32, 64)]),
      nn.ModuleList([self._upsample_module(64, 32), nn.Identity()]),
    ])
    self.dis_block_last = nn.Linear(7 * 7 * 64, 1)
    self.dis_shapes = [
      [-1, 32, 14, 14],
      [-1, 64, 7, 7],
    ]

    self.f_solver = broyden
    self.b_solver = broyden
    self.hook = None

  
  def _downsample_module(self, in_chan, out_chan):
    return conv3x3(in_chan, out_chan, stride=2)

  def _upsample_module(self, in_chan, out_chan):
    return nn.Sequential(OrderedDict([
      ('conv', nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)),
      ('upsample', nn.Upsample(scale_factor=2, mode='nearest'))]))


  def _forward_step(self, z, gen_injection, dis_injection):
    '''
    Compute one forward step F(z, x), where x is the injection.
    We do not really compute F(z, x), but rather solve z = F(z, x).
    But in the early stage of training, we could use this for pretraining.

    z: list of branch outputs
    injection_gen: The low-dim Gaussian noise.
    injection_dis: The real images.

    Returns: list of new branch outputs (z')
    '''
    num_gen_blocks = len(self.gen_blocks)
    num_dis_blocks = len(self.dis_blocks)
    assert len(z) == num_gen_blocks + num_dis_blocks

    # Step 1: Per-resolution residual block
    z_block = []
    for i in range(num_gen_blocks):
      s = self.gen_shapes[0]
      injection = self.gen_block_first(gen_injection).view(s) if i == 0 else 0
      z_block.append(self.gen_blocks[i](z[i] + injection))

    dis_injection = torch.cat((self.gen_block_last(z[num_gen_blocks - 1]), dis_injection))
    # print(dis_injection.shape)
    for j in range(num_dis_blocks):
      s = self.dis_shapes[0]
      injection = self.dis_block_first(dis_injection).view(s) if j == 0 else 0
      z_block.append(self.dis_blocks[j](z[num_gen_blocks + j] + injection))

    
    # Step 2: Multiscale fusion
    # In the original implementation there is a post-fuse layer
    # which is conv + relu
    z_fuse = []
    for i in range(num_gen_blocks):
      y = 0
      for j in range(num_gen_blocks):
        y += self.gen_fuse_blocks[j][i](z_block[j])
      z_fuse.append(y)

    for i in range(num_dis_blocks):
      y = 0
      for j in range(num_dis_blocks):
        y += self.dis_fuse_blocks[j][i](z_block[num_gen_blocks + j])
      z_fuse.append(y)

    return z_fuse
    

  def forward(self, x_noise, x_realimg, deq_mode=True, compute_jac_loss=None, **kwargs):
    '''
    Compute the forward pass with DEQ.

    x_noise: A batch of low-dim Gaussian noise
    x_realimg: A batch of real images
    deq_mode: If true then use DEQ. Otherwise, use NUM_LAYERS steps of _forward_step()

    '''
    size_noise = x_noise.shape[0]
    size_realimg = x_realimg.shape[0]
    num_gen_blocks = len(self.gen_blocks)
    num_dis_blocks = len(self.dis_blocks)
    device = self.device
    if compute_jac_loss is None:
      compute_jac_loss = self.config.get('compute_jac_loss', True)
    f_thres = self.config.get('f_thres', self.F_THRES)
    b_thres = self.config.get('b_thres', self.B_THRES)
    stop_mode = self.config.get('stop_mode', self.STOP_MODE)

    z_list = []
    for i in range(num_gen_blocks):
      s = self.gen_shapes[i]
      s[0] = size_noise
      z_list.append(torch.zeros(s).to(device))
    
    for j in range(num_dis_blocks):
      s = self.dis_shapes[j]
      s[0] = size_realimg + size_noise
      z_list.append(torch.zeros(s).to(device))

    z_vec = list2vec(z_list)
    shapes = self.gen_shapes + self.dis_shapes
    # The forward pass function
    func = lambda z: list2vec(self._forward_step(vec2list(z, shapes), x_noise, x_realimg))

    jac_loss = torch.tensor(0.0).to(device)

    # Main DEQ step
    if not deq_mode:
      num_layers = self.config.get('num_layers', self.NUM_LAYERS)
      for l in range(num_layers):
        z_vec = func(z_vec)
      new_z_vec = z_vec
      
      if self.training:
        if compute_jac_loss:
          z2 = z_vec.clone().detach().requires_grad_()
          new_z2 = func(z2)
          jac_loss = jac_loss_estimate(new_z2, z2)

    else:
      with torch.no_grad():
        result = self.f_solver(func, z_vec, threshold=f_thres,
                               stop_mode=stop_mode, name='forward')
        z_vec = result['result']
        new_z_vec = z_vec
      
      if self.training:
        new_z_vec = func(new_z_vec.requires_grad_())
        if compute_jac_loss:
          jac_loss = jac_loss_estimate(new_z_vec, z_vec)
        
        def backward_hook(grad):
          if self.hook is not None:
            self.hook.remove()
            if torch.cuda.is_available():
              torch.cuda.synchronize()
          result = self.b_solver(lambda y: autograd.grad(new_z_vec, z_vec, y, retain_graph=True)[0] + grad,
                                 torch.zeros_like(grad),
                                 threshold=b_thres, stop_mode=stop_mode, name='backward')
          return result['result']

        self.hook = new_z_vec.register_hook(backward_hook)

    # Compute the logits with the last fc block
    new_z_list = vec2list(new_z_vec, shapes)
    output = new_z_list[-1].flatten(start_dim=1)
    output = self.dis_block_last(output)
    return output, new_z_list, jac_loss


  def generate(self, x_noise, deq_mode=True):
    '''
    Generate fake images with the given noise
    '''
    # s = list(x_noise.shape)
    _, z_list, _ = self.forward(x_noise, torch.zeros((0, 1, 28, 28)).to(self.device),
                                deq_mode=deq_mode, compute_jac_loss=False)
    num_gen_blocks = len(self.gen_blocks)
    fake_imgs = self.gen_block_last(z_list[num_gen_blocks - 1])
    return fake_imgs

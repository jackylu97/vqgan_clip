import argparse
import math
from pathlib import Path
import sys

sys.path.append('./taming-transformers')

# from IPython import display
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import numpy as np
import os.path
from os import path
from urllib.request import Request, urlopen

from CLIP import clip
import kornia
import kornia.augmentation as K
from torch.utils.checkpoint import checkpoint

from matplotlib import pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar
import random
import gc

import re
from datetime import datetime

from base64 import b64encode

import warnings

import utils

warnings.filterwarnings('ignore')
torch.set_printoptions( sci_mode=False )

def noise_gen(shape, octaves=5):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    max_octaves = min(octaves, math.log(h)/math.log(2), math.log(w)/math.log(2))
    for i in reversed(range(max_octaves)):
        h_cur, w_cur = h // 2**i, w // 2**i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += torch.randn([n, c, h_cur, w_cur]) / 5
    return noise


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


# def replace_grad(fake, real):
#     return fake.detach() - real.detach() + real


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

replace_grad = ReplaceGrad.apply

clamp_with_grad = ClampWithGrad.apply
# clamp_with_grad = torch.clamp

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):

        input_normed = F.normalize(input.unsqueeze(1), dim=2)#(input / input.norm(dim=-1, keepdim=True)).unsqueeze(1)#
        embed_normed = F.normalize((self.embed).unsqueeze(0), dim=2)#(self.embed / self.embed.norm(dim=-1, keepdim=True)).unsqueeze(0)#

        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()

        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
    input_normed = F.normalize(input, dim=-1)
    target_normed = F.normalize(target, dim=-1)
    logits = input_normed @ target_normed.T * logit_scale
    if labels is None:
        labels = torch.arange(len(input), device=logits.device)
    return F.cross_entropy(logits, labels)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)

        for ii in range(self.cutn):
            size = int(min_size_width*torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound, 1.)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)

        return clamp_with_grad(cutouts, 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

class GaussianBlur2d(nn.Module):
    def __init__(self, sigma, window=0, mode='reflect', value=0):
        super().__init__()
        self.mode = mode
        self.value = value
        if not window:
            window = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
        if sigma:
            kernel = torch.exp(-(torch.arange(window) - window // 2)**2 / 2 / sigma**2)
            kernel /= kernel.sum()
        else:
            kernel = torch.ones([1])
        self.register_buffer('kernel', kernel)

    def forward(self, input):
        n, c, h, w = input.shape
        input = input.view([n * c, 1, h, w])
        start_pad = (self.kernel.shape[0] - 1) // 2
        end_pad = self.kernel.shape[0] // 2
        input = F.pad(input, (start_pad, end_pad, start_pad, end_pad), self.mode, self.value)
        input = F.conv2d(input, self.kernel[None, None, None, :])
        input = F.conv2d(input, self.kernel[None, None, :, None])
        return input.view([n, c, h, w])

class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""
    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer('biased', torch.zeros_like(tensor))
        self.register_buffer('average', torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer('accum', torch.tensor(1.))
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError('update() should only be called during training')

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average

import io
import base64
def image_to_data_url(img, ext):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=ext)
    img_byte_arr = img_byte_arr.getvalue()
    # ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    return prefix + base64.b64encode(img_byte_arr).decode('utf-8')


def update_random( seed, purpose ):
  if seed == -1:
    seed = random.seed()
    seed = random.randrange(1,99999)

  print( f'Using seed {seed} for {purpose}')
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  return seed

def clear_memory():
  gc.collect()

from typing import cast, Dict, Optional
from kornia.augmentation.base import IntensityAugmentationBase2D

class FixPadding(nn.Module):

    def __init__(self, module=None, threshold=1e-12, noise_frac=0.00 ):
        super().__init__()

        self.threshold = threshold
        self.noise_frac = noise_frac

        self.module = module

    def forward(self,input):

        dims = input.shape

        if self.module is not None:
            input = self.module(input + self.threshold)

        light = input.new_empty(dims[0],1,1,1).uniform_(0.,2.)

        mixed = input.view(*dims[:2],-1).sum(dim=1,keepdim=True)

        black = mixed < self.threshold
        black = black.view(-1,1,*dims[2:4]).type(torch.float)
        black = kornia.filters.box_blur( black, (5,5) ).clip(0,0.1)/0.1

        mean = input.view(*dims[:2],-1).sum(dim=2) / mixed.count_nonzero(dim=2)
        mean = ( mean[:,:,None,None] * light ).clip(0,1)

        fill = mean.expand(*dims)
        if 0 < self.noise_frac:
            rng = torch.get_rng_state()
            fill = fill + torch.randn_like(mean) * self.noise_frac
            torch.set_rng_state(rng)

        if self.module is not None:
            input = input - self.threshold

        return torch.lerp(input,fill,black)


class MyRandomNoise(IntensityAugmentationBase2D):
    def __init__(
        self,
        frac: float = 0.1,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0)
        self.frac = frac

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        noise = torch.FloatTensor(1).uniform_(0,self.frac)

        # generate pixel data without throwing off determinism of augs
        rng = torch.get_rng_state()
        noise = noise * torch.randn(shape)
        torch.set_rng_state(rng)

        return dict(noise=noise)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return input + params['noise'].to(input.device)

class MakeCutouts2(nn.Module):
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)

        for ii in range(self.cutn):
            size = int(min_size_width*torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound, 1.)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(cutout)

        return cutouts


class MultiClipLoss(nn.Module):
    def __init__(self, clip_models, text_prompt, normalize_prompt_weights, cutn, cut_pow=1., clip_weight=1., use_old_augs=False, simulate_old_cuts=False ):
        super().__init__()

        self.use_old_augs = use_old_augs
        self.simulate_old_cuts = simulate_old_cuts

        # Load Clip
        self.perceptors = []
        for cm in clip_models:
          c = clip.load(cm[0], jit=False)[0].eval().requires_grad_(False).to(device)
          self.perceptors.append( { 'res': c.visual.input_resolution, 'perceptor': c, 'weight': cm[1], 'prompts':[] } )
        self.perceptors.sort(key=lambda e: e['res'], reverse=True)

        # Make Cutouts
        self.cut_sizes = list(set([p['res'] for p in self.perceptors]))
        self.cut_sizes.sort( reverse=True )

        self.make_cuts = MakeCutouts2(self.cut_sizes[-1], cutn)

        # Get Prompt Embedings
        texts = [phrase.strip() for phrase in text_prompt.split("|")]
        if text_prompt == ['']:
          texts = []

        self.pMs = []

        prompts_weight_sum = 0
        parsed_prompts = []
        for prompt in texts:
          txt, weight, stop = parse_prompt(prompt)
          parsed_prompts.append( [txt,weight,stop] )
          prompts_weight_sum += max( weight, 0 )

        for prompt in parsed_prompts:
          txt, weight, stop = prompt
          clip_token = clip.tokenize(txt).to(device)

          if normalize_prompt_weights and 0 < prompts_weight_sum:
              weight /= prompts_weight_sum

          for p in self.perceptors:
            embed = p['perceptor'].encode_text(clip_token).float()
            embed_normed = F.normalize(embed.unsqueeze(0), dim=2)
            p['prompts'].append({'embed_normed':embed_normed,'weight':torch.as_tensor(weight, device=device),'stop':torch.as_tensor(stop, device=device)})

        # Prep Augments
        self.noise_fac = 0.1
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.1),
            FixPadding( nn.Sequential(
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='zeros'), # padding_mode=2
                K.RandomPerspective(0.2,p=0.4, ),
            )),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.15),
            MyRandomNoise(frac=self.noise_fac,p=1.),
        )

        self.clip_weight = clip_weight

    def prepare_cuts(self,img):
        cutouts = self.make_cuts(img)
        cutouts_out = []

        rng = torch.get_rng_state()

        for sz in self.cut_sizes:
            cuts = [resample(c, (sz,sz)) for c in cutouts]
            cuts = torch.cat(cuts, dim=0)
            cuts = clamp_with_grad(cuts,0,1)

            torch.set_rng_state(rng)
            cuts = self.augs(cuts)
            cuts = self.normalize(cuts)

            cutouts_out.append(cuts)

        return cutouts_out

    def forward( self, i, img ):
        cutouts = self.prepare_cuts( img )
        loss = []

        current_cuts = None
        currentres = 0

        for p in self.perceptors:
            if currentres != p['res']:
                currentres = p['res']
                current_cuts = cutouts[self.cut_sizes.index( currentres )]

            iii = p['perceptor'].encode_image(current_cuts).float()
            input_normed = F.normalize(iii.unsqueeze(1), dim=2)
            for prompt in p['prompts']:
                dists = input_normed.sub(prompt['embed_normed']).norm(dim=2).div(2).arcsin().pow(2).mul(2)
                dists = dists * prompt['weight'].sign()
                l = prompt['weight'].abs() * replace_grad(dists, torch.maximum(dists, prompt['stop'])).mean()
                loss.append(l * p['weight'])

        return loss

class MSEDecayLoss(nn.Module):
    def __init__(self, init_weight, mse_decay_rate, mse_epoches, mse_quantize ):
        super().__init__()

        self.init_weight = init_weight
        self.has_init_image = False
        self.mse_decay = init_weight / mse_epoches if init_weight else 0
        self.mse_decay_rate = mse_decay_rate
        self.mse_weight = init_weight
        self.mse_epoches = mse_epoches
        self.mse_quantize = mse_quantize

    @torch.no_grad()
    def set_target( self, z_tensor, model ):
        z_tensor = z_tensor.detach().clone()
        if self.mse_quantize:
            z_tensor = vector_quantize(z_tensor.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)#z.average
        self.z_orig = z_tensor

    def forward( self, i, z ):
        if self.is_active(i):
            return F.mse_loss(z, self.z_orig) * self.mse_weight / 2
        return 0

    def is_active(self, i):
        if not self.init_weight:
          return False
        if i <= self.mse_decay_rate and not self.has_init_image:
          return False
        return True

    @torch.no_grad()
    def step( self, i ):

        if i % self.mse_decay_rate == 0 and i != 0 and i < self.mse_decay_rate * self.mse_epoches:

            if self.mse_weight - self.mse_decay > 0 and self.mse_weight - self.mse_decay >= self.mse_decay:
              self.mse_weight -= self.mse_decay
            else:
              self.mse_weight = 0
            print(f"updated mse weight: {self.mse_weight}")

            return True

        return False

class TVLoss(nn.Module):
    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff**2 + y_diff**2 + 1e-8
        return diff.mean(dim=1).sqrt().mean()

torch.cuda.empty_cache()


import torch
import math

def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves( desired_shape, octaves=1, persistence=0.5):
    shape = torch.tensor(desired_shape)
    shape = 2 ** torch.ceil( torch.log2( shape ) )
    shape = shape.type(torch.int)

    max_octaves = int(min(octaves,math.log(shape[0])/math.log(2), math.log(shape[1])/math.log(2)))
    res = torch.floor( shape / 2 ** max_octaves).type(torch.int)

    noise = torch.zeros(list(shape))
    frequency = 1
    amplitude = 1
    for _ in range(max_octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise[:desired_shape[0],:desired_shape[1]]

def rand_perlin_rgb( desired_shape, amp=0.1, octaves=6 ):
  r = rand_perlin_2d_octaves( desired_shape, octaves )
  g = rand_perlin_2d_octaves( desired_shape, octaves )
  b = rand_perlin_2d_octaves( desired_shape, octaves )
  rgb = ( torch.stack((r,g,b)) * amp + 1 ) * 0.5
  return rgb.unsqueeze(0).clip(0,1).to(device)


def pyramid_noise_gen(shape, octaves=5, decay=1.):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    max_octaves = int(min(math.log(h)/math.log(2), math.log(w)/math.log(2)))
    if octaves is not None and 0 < octaves:
      max_octaves = min(octaves,max_octaves)
    for i in reversed(range(max_octaves)):
        h_cur, w_cur = h // 2**i, w // 2**i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += ( torch.randn([n, c, h_cur, w_cur]) / max_octaves ) * decay**( max_octaves - (i+1) )
    return noise

def rand_z(model, toksX, toksY):
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    return z


def make_rand_init( mode, model, perlin_octaves, perlin_weight, pyramid_octaves, pyramid_decay, toksX, toksY, f ):

  if mode == 'VQGAN ZRand':
    return rand_z(model, toksX, toksY)
  elif mode == 'Perlin Noise':
    rand_init = rand_perlin_rgb((toksY * f, toksX * f), perlin_weight, perlin_octaves )
    z, *_ = model.encode(rand_init * 2 - 1)
    return z
  elif mode == 'Pyramid Noise':
    rand_init = pyramid_noise_gen( (1,3,toksY * f, toksX * f), pyramid_octaves, pyramid_decay).to(device)
    rand_init = ( rand_init * 0.5 + 0.5 ).clip(0,1)
    z, *_ = model.encode(rand_init * 2 - 1)
    return z


#@title Set Display Rate
#@markdown If `use_automatic_display_schedule` is enabled, the image will be output frequently at first, and then more spread out as time goes on. Turn this off if you want to specify the display rate yourself.
use_automatic_display_schedule = True #@param {type:'boolean'}
display_every = 50 #@param {type:'number'}

def should_checkin(i):
  if i == max_iter:
    return True

  if not use_automatic_display_schedule:
    return i % display_every == 0

  schedule = [[100,25],[500,50],[1000,100],[2000,200]]
  for s in schedule:
    if i <= s[0]:
      return i % s[1] == 0
  return i % 500 == 0



#@markdown What do you want to see?
text_prompt = 'The Enchanted Garden by James Gurney'#@param {type:'string'}
gen_seed = -1#@param {type:'number'}
#@markdown - If you want to keep starting from the same point, set `gen_seed` to a positive number. `-1` will make it random every time.
init_image = ''#@param {type:'string'}
width = 600#@param {type:'number'}
height = 400#@param {type:'number'}
max_iter = 501 #@param {type:'number'}


#@markdown There are different ways of generating the random starting point, when not using an init image. These influence how the image turns out. The default VQGAN ZRand is good, but some models and subjects may do better with perlin or pyramid noise.
rand_init_mode = 'VQGAN ZRand'#@param [ "VQGAN ZRand", "Perlin Noise", "Pyramid Noise"]
perlin_octaves = 7#@param {type:"slider", min:1, max:8, step:1}
perlin_weight = 0.22#@param {type:"slider", min:0, max:1, step:0.01}
pyramid_octaves = 5#@param {type:"slider", min:1, max:8, step:1}
pyramid_decay = 0.99#@param {type:"slider", min:0, max:1, step:0.01}
ema_val = 0.99

#@markdown How many slices of the image should be sent to CLIP each iteration to score? Higher numbers are better, but cost more memory. If you are running into memory issues try lowering this value.
cut_n = 64 #@param {type:'number'}

#@markdown One clip model is good. Two is better? You may need to reduce the number of cuts to support having more than one CLIP model. CLIP is what scores the image against your prompt and each model has slightly different ideas of what things are.
#@markdown - `ViT-B/32` is fast and good and what most people use to begin with

clip_model = 'ViT-B/32' #@param ["ViT-B/16", "ViT-B/32", "RN50x16", "RN50x4"]
clip_model2 ='None' #@param ["None","ViT-B/16", "ViT-B/32", "RN50x16", "RN50x4"]
if clip_model2 == "None":
    clip_model2 = None
clip1_weight = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}

#@markdown Picking a different VQGAN model will impact how an image generates. Think of this as giving the generator a different set of brushes and paints to work with. CLIP is still the "eyes" and is judging the image against your prompt but using different brushes will make a different image.
#@markdown - `vqgan_imagenet_f16_16384` is the default and what most people use
vqgan_model = 'vqgan_imagenet_f16_16384'#@param [ "vqgan_imagenet_f16_1024", "vqgan_imagenet_f16_16384", "vqgan_openimages_f8_8192", "coco", "faceshq","wikiart_1024", "wikiart_16384", "sflckr"]

#@markdown Learning rates greatly impact how quickly an image can generate, or if an image can generate at all. The first learning rate is only for the first 50 iterations. The epoch rate is what is used after reaching the first mse epoch.
#@markdown You can try lowering the epoch rate while raising the initial learning rate and see what happens
learning_rate = 0.2#@param {type:'number'}
learning_rate_epoch = 0.2#@param {type:'number'}
#@markdown How much should we try to match the init image, or if no init image how much should we resist change after reaching the first epoch?
mse_weight = 0.5 #@param {type:'number'}
#@markdown Adding some TV may make the image blurrier but also helps to get rid of noise. A good value to try might be 0.1.
tv_weight = 0.0 #@param {type:'number'}
#@markdown Should the total weight of the text prompts stay in the same range, relative to other loss functions?
normalize_prompt_weights = True #@param {type:'boolean'}

#@markdown Enabling the EMA tensor will cause the image to be slower to generate but may help it be more cohesive.
#@markdown This can also help keep the final image closer to the init image, if you are providing one.
use_ema_tensor = False #@param {type:'boolean'}

#@markdown If you want to generate a video of the run, you need to save the frames as you go. The more frequently you save, the longer the video but the slower it will take to generate.
save_art_output = True #@param {type:'boolean'}
save_frames_for_video = False #@param {type:'boolean'}
save_frequency_for_video = 3  #@param {type:'number'}


#@markdown  ----
#@markdown  I'd love to see what you can make with my notebook. Tweet me your art [@remi_durant](https://twitter.com/remi_durant)!

output_as_png = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)
print('using prompts: ', text_prompt)

clear_memory()

model = utils.get_vqgan_model( vqgan_model )

if clip_model2:
  clip_models = [[clip_model, clip1_weight], [clip_model2, 1. - clip1_weight]]
else:
  clip_models = [[clip_model, 1.0]]
print(clip_models)

clip_loss = MultiClipLoss( clip_models, text_prompt, normalize_prompt_weights=normalize_prompt_weights, cutn=cut_n)

seed = update_random( gen_seed, 'image generation')

# Make Z Init
z = 0

f = 2**(model.decoder.num_resolutions - 1)
toksX, toksY = math.ceil( width / f), math.ceil( height / f)

print(f'Outputing size: [{toksX*f}x{toksY*f}]')

has_init_image = (init_image != "")
if has_init_image:
    if 'http' in init_image:
      req = Request(init_image, headers={'User-Agent': 'Mozilla/5.0'})
      img = Image.open(urlopen(req))
    else:
      img = Image.open(init_image)

    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((toksX * f, toksY * f), Image.LANCZOS)
    pil_image = TF.to_tensor(pil_image)
    #if args.use_noise:
    #  pil_image = pil_image + args.use_noise * torch.randn_like(pil_image)
    z, *_ = model.encode(pil_image.to(device).unsqueeze(0) * 2 - 1)
    del pil_image
    del img

else:
    z = make_rand_init( rand_init_mode, model, perlin_octaves, perlin_weight, pyramid_octaves, pyramid_decay, toksX, toksY, f )

z = EMATensor(z, ema_val)

opt = optim.Adam( z.parameters(), lr=learning_rate, weight_decay=0.00000000)

mse_loss = MSEDecayLoss( mse_weight, mse_decay_rate=50, mse_epoches=5, mse_quantize=True )
mse_loss.set_target( z.tensor, model )
mse_loss.has_init_image = has_init_image

tv_loss = TVLoss()


losses = []
mb = master_bar(range(1))
gnames = ['losses']

# mb.names=gnames
# mb.graph_fig, axs = plt.subplots(1, 1) # For custom display
# mb.graph_ax = axs
# mb.graph_out = display.display(mb.graph_fig, display_id=True)

## optimizer loop

def synth(z, quantize=True, scramble=True):
    z_q = 0
    if quantize:
      z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    else:
      z_q = z.model

    out = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    return out

@torch.no_grad()
def checkin(i, z, out_pil, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')

    # display_format='png' if output_as_png else 'jpg'
    # pil_data = image_to_data_url(out_pil, display_format)

    # display.display(display.HTML(f'<img src="{pil_data}" />'))

def should_save_for_video(i):
    return save_frames_for_video and i % save_frequency_for_video

def train(i):
    global opt
    global z
    opt.zero_grad( set_to_none = True )

    out = checkpoint( synth, z.tensor )

    lossAll = []
    lossAll += clip_loss( i,out )

    if 0 < mse_weight:
      msel = mse_loss(i,z.tensor)
      if 0 < msel:
        lossAll.append(msel)

    if 0 < tv_weight:
      lossAll.append(tv_loss(out)*tv_weight)

    loss = sum(lossAll)
    loss.backward()

    if should_checkin(i) or should_save_for_video(i):
        with torch.no_grad():
            if use_ema_tensor:
                out = synth( z.average )

            pil = TF.to_pil_image(out[0].cpu())

            if should_checkin(i):
                checkin(i, z, pil, lossAll)
                if save_art_output:
                    utils.save_output(pil, text_prompt, seed, i)

            if should_save_for_video(i):
                pil.save(f'steps/step{i//save_frequency_for_video:04}.png')

    # update graph
    losses.append(loss)
    x = range(len(losses))
    mb.update_graph( [[x,losses]] )

    opt.step()
    if use_ema_tensor:
      z.update()

i = 0
try:
  with tqdm() as pbar:
    while True and i <= max_iter:

      if i % 200 == 0:
        clear_memory()

      train(i)

      with torch.no_grad():
          if mse_loss.step(i):
              print('Reseting optimizer at mse epoch')

              if mse_loss.has_init_image and use_ema_tensor:
                mse_loss.set_target(z.average,model)
              else:
                mse_loss.set_target(z.tensor,model)

              # Make sure not to spike loss when mse_loss turns on
              if not mse_loss.is_active(i):
                z.tensor = nn.Parameter(mse_loss.z_orig.clone())
                z.tensor.requires_grad = True

              if use_ema_tensor:
                z = EMATensor(z.average, ema_val)
              else:
                z = EMATensor(z.tensor, ema_val)

              opt = optim.Adam(z.parameters(), lr=learning_rate_epoch, weight_decay=0.00000000)

      i += 1
      pbar.update()

except KeyboardInterrupt:
    pass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

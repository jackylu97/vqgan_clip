import os
import requests
import sys

sys.path.append('./taming-transformers')
from taming.models import cond_transformer, vqgan
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import re
from datetime import datetime


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vqgan_path = "./models/VQGAN/"
if not os.path.exists(vqgan_path):
    os.makedirs(vqgan_path)

output_path = "./outputs/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

#@markdown When saving the images, how much should be included in the name?
include_full_prompt_in_filename = False #@param {type:'boolean'}
shortname_limit = 50 #@param {type: 'number'}
filename_limit = 250

model_download={
  "vqgan_imagenet_f16_1024":
      [["vqgan_imagenet_f16_1024.yaml", "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"],
      ["vqgan_imagenet_f16_1024.ckpt", "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1"]],
  "vqgan_imagenet_f16_16384":
      [["vqgan_imagenet_f16_16384.yaml", "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"],
      ["vqgan_imagenet_f16_16384.ckpt", "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"]],
  "vqgan_openimages_f8_8192":
      [["vqgan_openimages_f8_8192.yaml", "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"],
      ["vqgan_openimages_f8_8192.ckpt", "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1"]],
  "coco":
      [["coco_first_stage.yaml", "http://batbot.tv/ai/models/VQGAN/coco_first_stage.yaml"],
      ["coco_first_stage.ckpt", "http://batbot.tv/ai/models/VQGAN/coco_first_stage.ckpt"]],
  "faceshq":
      [["faceshq.yaml", "https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT"],
      ["faceshq.ckpt", "https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt"]],
  "wikiart_1024":
      [["wikiart_1024.yaml", "http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml"],
      ["wikiart_1024.ckpt", "http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt"]],
  "wikiart_16384":
      [["wikiart_16384.yaml", "http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml"],
      ["wikiart_16384.ckpt", "http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt"]],
  "sflckr":
      [["sflckr.yaml", "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1"],
      ["sflckr.ckpt", "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1"]],
  }

loaded_model = None
loaded_model_name = None
def dl_vqgan_model(image_model):
    for curl_opt in model_download[image_model]:
        modelpath = f'{vqgan_path}{curl_opt[0]}'
        if not os.path.exists(modelpath):
            print(f'downloading {modelpath} from {curl_opt[1]}')
            response = requests.get(curl_opt[1])
            with open(modelpath, 'wb') as f:
                f.write(response.content)
        else:
            print(f'found existing {curl_opt[0]}')

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

def get_vqgan_model(image_model):
    global loaded_model
    global loaded_model_name
    if loaded_model is None or loaded_model_name != image_model:
        dl_vqgan_model(image_model)

        print(f'loading {image_model} vqgan checkpoint')

        vqgan_config= vqgan_path + model_download[image_model][0][0]
        vqgan_checkpoint= vqgan_path + model_download[image_model][1][0]
        print('vqgan_config',vqgan_config)
        print('vqgan_checkpoint',vqgan_checkpoint)

        model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
        if image_model == 'vqgan_openimages_f8_8192':
            model.quantize.e_dim = 256
            model.quantize.n_e = model.quantize.n_embed
            model.quantize.embedding = model.quantize.embed

        loaded_model = model
        loaded_model_name = image_model

    return loaded_model

def slugify(value):
    value = str(value)
    value = re.sub(r':([-\d.]+)', ' [\\1]', value)
    value = re.sub(r'[|]','; ',value)
    value = re.sub(r'[<>:"/\\|?*]', ' ', value)
    return value

def get_filename(text, seed, i, ext):
    if ( not include_full_prompt_in_filename ):
        text = re.split(r'[|:;]',text, 1)[0][:shortname_limit]
    text = slugify(text)

    now = datetime.now()
    t = now.strftime("%y%m%d%H%M")
    if i is not None:
        data = f'; r{seed} i{i} {t}{ext}'
    else:
        data = f'; r{seed} {t}{ext}'

    return text[:filename_limit-len(data)] + data

def save_output(pil, text, seed, i):
    fname = get_filename(text,seed,i,'.png')
    pil.save(output_path + fname)

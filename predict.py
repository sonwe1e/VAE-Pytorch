import cv2
import torch
import argparse
import yaml
from models import *

device = 'cuda:0'
# 加载模型
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                dest="filename",
                metavar='FILE',
                help =  'path to the config file',
                default='configs/vae.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)
model = vae_models[config['model_params']['name']](**config['model_params']).to(config['trainer_params']['gpus'])


# 生成隐变量数据
b = 10
model.load_state_dict(torch.load('checkpoints/VAE_best.pth'))
model.eval()
samples = model.sample(10, device)
for i in range(b):
    cv2.imwrite(f'{i:02d}.png', samples[i].permute(1, 2, 0).cpu().detach().numpy()*255)
import cv2
import yaml
import torch
import argparse
from models import *
from einops import rearrange

def predict(config):
    model = vae_models[config['model_params']['name']](**config['model_params']).to(config['trainer_params']['gpus'])

    # 生成隐变量数据
    b = 64
    model.load_state_dict(torch.load(config['logging_params']['ckpt_save_dir']+'best.pth'))
    model.eval()
    samples = model.sample(b, config['trainer_params']['gpus'])
    samples = samples.permute(0, 2, 3, 1).cpu().detach().numpy()*255
    samples = rearrange(samples, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=8)
    cv2.imwrite(config['logging_params']['img_save_dir']+'sample.png', samples)
    samples = model.sample(b, config['trainer_params']['gpus'],
                           z=torch.randn(1, config['model_params']['latent_dim'])+torch.arange(-b, b, 2).unsqueeze(1) / b)
    samples = samples.permute(0, 2, 3, 1).cpu().detach().numpy()*255
    samples = rearrange(samples, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=8)
    cv2.imwrite(config['logging_params']['img_save_dir']+'continuous.png', samples)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)
    predict(config)
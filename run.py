import torch
from tqdm import tqdm
import yaml
import argparse
from models import *
import wandb
import multiprocessing as mp
from dataset import get_face_data
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)
            
    run = wandb.init(
        project='VAE-Package',
        name = config['model_params']['name'],
        config=config
    )
    model = vae_models[config['model_params']['name']](**config['model_params']).to(config['trainer_params']['gpus'])

    print('Getting data...')
    train_loader, val_loader = get_face_data(
        batch_size=config['data_params']['batch_size'],
        num_workers=config['data_params']['num_workers'], 
        patch_size=config['data_params']['patch_size'])
    print('Data loaded.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['exp_params']['learning_rate'])
    
    best_loss = 10000
    max_epochs = config['trainer_params']['max_epochs']
    for epoch in tqdm(range(1, max_epochs)):
        train_loss = 0
        train_rec_loss = 0
        train_kdl_loss = 0
        valid_loss = 0
        valid_rec_loss = 0
        valid_kdl_loss = 0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for b, (img, attrib) in loop:
            img = img.to(config['trainer_params']['gpus'])
            optimizer.zero_grad()
            img_args = model.forward(img, attrib=attrib)
            loss_args = model.loss_function(
                *img_args,
                M_N = config['exp_params']['kld_weight'],
            )
            loss_args['loss'].backward()
            optimizer.step()
            train_loss += loss_args['loss'].item() / len(train_loader)
            train_rec_loss += loss_args['reconstruction_loss'].item() / len(train_loader)
            train_kdl_loss += loss_args['kdl_loss'].item() / len(train_loader)
            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(train_loss = loss_args['loss'].cpu().detach().numpy())
        wandb.log({'train_loss': train_loss, 'train_rec_loss': train_rec_loss, 'train_kdl_loss': train_kdl_loss})
        
        loop = tqdm(enumerate(val_loader), total =len(val_loader))
        with torch.no_grad():
            model.eval()
            for b, (img, attrib) in loop:
                img = img.to(config['trainer_params']['gpus'])
                img_args = model.forward(img, attrib=attrib)
                loss_args = model.loss_function(
                *img_args,
                M_N = config['exp_params']['kld_weight'],
                )
                valid_loss += loss_args['loss'].item() / len(val_loader)
                valid_rec_loss += loss_args['reconstruction_loss'].item() / len(val_loader)
                valid_kdl_loss += loss_args['kdl_loss'].item() / len(val_loader)
                loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
                loop.set_postfix(valid_loss = loss_args['loss'].cpu().detach().numpy())
            wandb.log({'valid_loss': valid_loss, 'valid_rec_loss': valid_rec_loss, 'valid_kdl_loss': valid_kdl_loss})
            # samples = model.sample(10, config['trainer_params']['gpus'])
            if valid_loss < best_loss:
                torch.save(model.state_dict(), f'./checkpoints/{config["model_params"]["name"]}_best.pth')
                best_loss = valid_loss
                wandb.log({'best_loss': best_loss})
    torch.save(model.state_dict(), f'./checkpoints/{config["model_params"]["name"]}_last.pth')
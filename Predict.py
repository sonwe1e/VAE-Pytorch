import cv2
import torch
from collections import OrderedDict
from Model import CVAE
from utils import label2onehot

model = CVAE().cuda(3)

old_state_dict = torch.load('checkpoints/cvae-epoch=212-val_loss=13419.76.ckpt')['state_dict']
new_state_dict = OrderedDict()

for k, v in old_state_dict.items():
    name = k[6:] # remove `model.`
    new_state_dict[name] = v
    
model.load_state_dict(new_state_dict)

test_z = torch.randn(10, 64, device='cuda:3')
test_class = torch.ones(10, 1, device='cuda:3', dtype=torch.int64) * 7
test_class = label2onehot(test_class, 10)
test_z = torch.cat([test_z, test_class], dim=1)
output = model.decoder(test_z)

output = output.cpu().detach().numpy().reshape(10, 28, 28)
for i in range(10):
    cv2.imwrite(f'8/{i}.png', 255*output[i])
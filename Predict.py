import cv2
import torch
from collections import OrderedDict
from MNISTModel import CVAE
from FaceModel import FaceVAE
from utils import label2onehot

model = FaceVAE().cuda(2)

old_state_dict = torch.load('checkpoints/cvae-epoch=40-val_loss=224579.20.ckpt')['state_dict']
new_state_dict = OrderedDict()

for k, v in old_state_dict.items():
    name = k[6:] # remove `model.`
    new_state_dict[name] = v
    
model.load_state_dict(new_state_dict)

test_z = torch.randn((10, 120, 1, 1), device='cuda:2')
test_class = torch.ones((10, 40, 1, 1), device='cuda:2', dtype=torch.int64) - 2
test_class[:, 20, :, :] = 1
# test_class = label2onehot(test_class, 10)
test_z = torch.cat([test_z, test_class], dim=1)
output = model.decoder(test_z)

output = output.permute(0, 2, 3, 1).cpu().detach().numpy()
print(output.shape)
for i in range(10):
    cv2.imwrite(f'8/{i}.png', 255*output[i])
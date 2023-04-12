import cv2
import torch
from collections import OrderedDict
from MNISTModel import CVAE
from FaceModel import FaceVAE
from utils import label2onehot

device = 'cuda:0'
# 加载模型
model = FaceVAE().to(device)
old_state_dict = torch.load('/media/sonwe1e/ade2775b-8298-bf46-9e1e-aa116abe4660/CVAE/checkpoints/cvae-epoch=102-val_loss=377410.28.ckpt', map_location=device)['state_dict']
new_state_dict = OrderedDict()
for k, v in old_state_dict.items():
    name = k[6:] # 移除 `model.` 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()


b = 2000
# 生成隐变量数据
test_z = torch.zeros((1, 120, 1, 1), device=device).repeat(b, 1, 1, 1) # 生成随机噪声
increments = torch.arange(-2, 2, 0.004, device=device)

# 循环遍历每个特征，增加对应的值
for i, inc in enumerate(increments):
    test_z[i] += inc
test_class = torch.sign(torch.randn((1, 40, 1, 1), device=device)-0.7).repeat(b, 1, 1, 1) # 生成随机类别
# test_class[:, 2, :, :] = 1 # 生成指定类别
test_z = torch.cat([test_z, test_class], dim=1) # 拼接噪声和类别

# 生成隐变量解码预测结果
batch = 100
for i in range(1, int(b / batch)):
    output = model.decoder(test_z[i*batch:(i+1)*batch,...])
    output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

# 保存预测结果到图像文件
    for j in range(batch):
        cv2.imwrite(f'Example/Prediction/{batch*(i-1)+j:04d}.png', 255*output[j])

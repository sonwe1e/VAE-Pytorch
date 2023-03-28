import cv2
import torch
from collections import OrderedDict
from MNISTModel import CVAE
from FaceModel import FaceVAE
from utils import label2onehot

device = 'cuda:3'
# 加载模型
model = FaceVAE().to(device)
old_state_dict = torch.load('checkpoints/cvae-epoch=60-val_loss=378360.19.ckpt', map_location=device)['state_dict']
new_state_dict = OrderedDict()
for k, v in old_state_dict.items():
    name = k[6:] # 移除 `model.` 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# 生成隐变量数据
test_z = torch.randn((10, 120, 1, 1), device=device) # 生成随机噪声
test_class = torch.sign(torch.randn((10, 40, 1, 1), device=device)-4) # 生成随机类别
test_class[:, 2, :, :] = 1 # 生成指定类别
test_z = torch.cat([test_z, test_class], dim=1) # 拼接噪声和类别

# 生成隐变量解码预测结果
output = model.decoder(test_z)
output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

# 利用编码器将预测结果再次编码
test = cv2.imread('Example/test.png')

test = cv2.resize(test, (256, 256))
cv2.imwrite('Example/test.png', test)
test = torch.from_numpy(test/256).permute(2, 0, 1).unsqueeze(0).float().to(device)
print(test.shape, test_class[0:,:,0,0].shape)
test = model(test, test_class[0:1,:,0,0])
# print(test.shape)
cv2.imwrite('Example/test2.png', 255*test[0][0].permute(1, 2, 0).cpu().detach().numpy())

# 保存预测结果到图像文件
for i in range(10):
    cv2.imwrite(f'Example/Prediction/{i}.png', 255*output[i])

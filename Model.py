import torch
import torch.nn as nn
from utils import label2onehot

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=0):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_dim+num_classes, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leakyrelu(self.l1(x))
        x = self.leakyrelu(self.l2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, output_dim=784, num_classes=10):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(z_dim+num_classes, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.l1(x))
        x = self.leakyrelu(self.l2(x))
        x = self.Sigmoid(x)
        return x


class CVAE(nn.Module):
    def __init__(self, num_classes=10):
        super(CVAE, self).__init__()
        self.encoder = Encoder(num_classes=10)
        self.decoder = Decoder(num_classes=10)
        self.fc_mu = nn.Linear(512, 64)
        self.fc_var = nn.Linear(512, 64)

    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs, y):
        # 将输入的图像进行编码, 得到均值和方差
        inputs = inputs.reshape(inputs.size(0), -1)
        x = torch.cat([inputs, y], dim=1)
        x = self.encoder(x)
        mu = self.fc_mu(x)  # 均值
        logvar = self.fc_var(x)  # 方差
        z = self.reparameter(mu, logvar)  # 重参数化
        z = torch.cat([z, y], dim=1)
        x = self.decoder(z)
        return [x, inputs, mu, logvar]

if __name__ == '__main__':
    net = CVAE().cuda(4)
    x = torch.randn(2, 1, 28, 28).cuda(4)
    y = torch.tensor([[0], [1]]).cuda(4)
    y = label2onehot(y, 10)
    y = net(x, y)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)
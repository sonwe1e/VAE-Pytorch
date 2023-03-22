import torch
import timm
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x + self.alpha * res)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.f = BasicBlock(out_channel, out_channel)
        
    def forward(self, x):
        x = self.up(x)
        x = self.f(x)
        return x

class FaceVAE(nn.Module):
    def __init__(self, Attr=40):
        super(FaceVAE, self).__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, features_only=True)
<<<<<<< HEAD
        channel_list = [Attr*2, 256, 128, 64, 32, 16]
        self.decoder = nn.Sequential(
            nn.Conv2d(Attr*2, channel_list[1], 1, 1, 0),
=======
        channel_list = [Attr*4, 256, 128, 64, 32, 16]
        self.decoder = nn.Sequential(
            nn.Conv2d(Attr*4, channel_list[1], 1, 1, 0),
>>>>>>> 28bb9bc1bc376b70c11fdafe082c2419e2de37ec
            nn.PixelShuffle(2),
            nn.Conv2d(channel_list[1] // 4, channel_list[1], 3, 1, 1),
            nn.InstanceNorm2d(channel_list[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(4),
            nn.Conv2d(channel_list[1] // 16, channel_list[1], 3, 1, 1),
            nn.InstanceNorm2d(channel_list[1]),
            nn.LeakyReLU(0.2, inplace=True),
            UpBlock(channel_list[1], channel_list[1]),
            UpBlock(channel_list[1], channel_list[2]),
            UpBlock(channel_list[2], channel_list[3]),
            UpBlock(channel_list[3], channel_list[4]),
            UpBlock(channel_list[4], channel_list[5]),
            nn.Conv2d(channel_list[5], 3, 1, 1, 0),
            nn.Tanh())
        self.mu_conv = nn.Conv2d(512, Attr*3, 8, 8, 0)
        self.logvar_conv = nn.Conv2d(512, Attr*3, 8, 8, 0)
        
    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data, y):
        x = self.encoder(data)[-1]
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        Attr = y.unsqueeze(2).unsqueeze(3).repeat(1, 1, mu.shape[2], mu.shape[3])
        z = self.reparameter(mu, logvar)
        z = torch.cat([z, Attr], dim=1)
        z = self.decoder(z)
        return z, data, mu, logvar
    
if __name__ == '__main__':
    model = FaceVAE()
    x = torch.rand(2, 3, 256, 256)
    y = torch.rand(2, 40)
    out = model(x, y)
    for i in out:
        print(i.shape)
        
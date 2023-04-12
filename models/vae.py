import torch
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None,
                 **kwargs):
        super().__init__()

        # 定义存放模型的容器
        modules = []
        # 定义中间卷积层的通道数
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.latent_dim = latent_dim
        self.last_dim = hidden_dims[-1]

        # 构建编码器 结构为(conv(3, 2, 1)->bn->leakyrelu)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # 定义均值和方差的线性层
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        modules = []
        
        # 对均值和方差进行解码
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # 反向通道数顺序
        hidden_dims.reverse()
        # 定义解码器 结构为(Deconv(3, 2, 1)->bn->leakyrelu)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
    def encode(self, input):
        """
        对输入的图像进行编码，得到均值和方差
        :param input: 输入的维度为 [N x C x H x W]
        :return: 输出为均值和方差构成的列表
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        通过均值和方差进行解码，得到图像
        :param z: 输入为维度 [B x D] 的编码
        :return: 输出图像 [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.last_dim, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        将均值和方差进行重参数化，使其满足正态分布
        :param mu: 隐变量的均值 [B x D]
        :param logvar: 隐变量的标准差 [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]
    
    def loss_function(self, *args, **kwargs):
        """
        计算标准VAE的损失，包括重构误差以及KLD误差
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args: 输入通常为 [重建图像, 输入图像, 均值, 标准差]
        :param kwargs: 一般是KLD损失前的权重
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruction_loss':recons_loss.detach(), 'kdl_loss':-kld_loss.detach()}

    def sample(self, num_samples, current_device, **kwargs):
        """
        从编码的分布中采样并重建图像
        :param num_samples: 重建样本的数量
        :param current_device: 进行采样的网络
        :return: 图像
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x, **kwargs):
        """
        返回进行重建的图像
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
    
if __name__ == '__main__':
    model = VAE(3, 128, hidden_dims=None).cuda()
    x = torch.rand(2, 3, 64, 64, device='cuda')
    out = model(x)
    for i in out:
        print(i.shape)
        
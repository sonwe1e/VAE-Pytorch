import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tiger import Tiger

# Construct two dimension autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, mid_dim=1536, out_dim=2048):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, 784),
        )

        self.mu = nn.Linear(out_dim, out_dim)
        self.logvar = nn.Linear(out_dim, out_dim)

    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        h = self.reparameter(mu, logvar)
        x = self.decoder(h)
        return x, h, mu, logvar


# 导入 MNIST 数据
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

# 定义测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 定义模型
model = AutoEncoder().cuda(5)
# 定义损失函数
criterion = nn.MSELoss()
# 定义优化器
optimizer = Tiger(model.parameters(), lr=2e-4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

label_loss = 100
# 训练模型
for epoch in range(100):
    test_loss = 0
    for x, y in train_loader:
        x, y = x.view(x.size(0), -1).cuda(5), y.cuda(5).float()
        out, h, mu, logvar = model(x)
        loss = criterion(out, x) - torch.mean(1 - logvar.pow(2)).pow(2)  + criterion(torch.mean(mu, dim=1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

    # 测试模型
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.view(x.size(0), -1).cuda(5), y.cuda(5).float()
            out, h, mu, logvar = model(x)
            loss = criterion(out, x) - torch.mean(1 - logvar.pow(2)).pow(2)  + criterion(torch.mean(mu, dim=1), y)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        print('Epoch: {}, Test Loss: {:.4f}'.format(epoch, test_loss))
        if test_loss < label_loss:
            label_loss = test_loss
            torch.save(model.state_dict(), 'model.pth')

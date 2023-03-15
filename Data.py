# 导入 MNIST 数据
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义对数据的预处理
transform = transforms.Compose([transforms.ToTensor()])

# 训练集
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transform,
                               download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transform)

# 定义训练数据的迭代器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=512, num_workers=40,
                          shuffle=True)

# 定义测试数据的迭代器
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=512, num_workers=40,
                         shuffle=False)


def get_data():
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_data()
    print(next(iter(train_loader))[1].shape)
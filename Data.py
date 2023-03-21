import os
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_attr():
    # 加载celeba数据集的人脸属性
    with open('./data/list_attr_celeba.txt', 'r') as f:
        attr = f.readlines()
        
    # 去掉第一二行
    attr = attr[2:]
    attr = [i.strip().split() for i in attr]
    img_name = [i[0] for i in attr]
    attr_label = [i[1:] for i in attr]
    # 将人脸属性转换为int类型
    attr_label = [[int(j) for j in i] for i in attr_label]
    return img_name, attr_label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

def get_digit_data():
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
                            batch_size=512, num_workers=16,
                            shuffle=True)

    # 定义测试数据的迭代器
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=512, num_workers=16,
                            shuffle=False)

    return train_loader, test_loader


class FaceDataset(Dataset):
    def __init__(self, transform=None, mode='Train') -> None:
        super().__init__()
        self.ImgsPath = './data/img_align_celeba/'
        self.AttributePath = './data/list_attr_celeba.txt'
        self.ImgName, AttrLabel = get_attr()
        self.AttrLabel = torch.tensor(AttrLabel)
        self.transform = transform
        if mode == 'Train':
            self.ImgName = self.ImgName[:162770]
            self.AttrLabel = self.AttrLabel[:162770]
            
        elif mode == 'Valid':
            self.ImgName = self.ImgName[162770:]
            self.AttrLabel = self.AttrLabel[162770:]
        
    def __len__(self):
        return len(self.ImgName)
    
    def __getitem__(self, index):
        ImgPath = os.path.join(self.ImgsPath, self.ImgName[index])
        Img = cv2.imread(ImgPath)
        if self.transform:
            Img = self.transform(Img)
            
        AttrLabel = self.AttrLabel[index]
        return Img, AttrLabel.float()
    
def get_face_data():
    train_data = FaceDataset(transform, 'Train')
    valid_data = FaceDataset(transform, 'Valid')
    TrainLoader = DataLoader(dataset=train_data,
                            batch_size=128, num_workers=16,
                            shuffle=True)
    ValidLoader = DataLoader(dataset=valid_data,
                            batch_size=512, num_workers=16,
                            shuffle=False)
    return TrainLoader, ValidLoader

if __name__ == '__main__':
    train_loader, test_loader = get_face_data()
    for x, y in train_loader:
        print(y[0])
        assert 0
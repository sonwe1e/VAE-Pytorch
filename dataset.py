import os
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import get_attr

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
    
def get_face_data(batch_size=64, num_workers=16, patch_size=64):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])    
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    train_data = FaceDataset(train_transform, 'Train')
    valid_data = FaceDataset(valid_transform, 'Valid')
    TrainLoader = DataLoader(dataset=train_data,
                            batch_size=batch_size, num_workers=num_workers,
                            shuffle=True, persistent_workers=True)
    ValidLoader = DataLoader(dataset=valid_data,
                            batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, persistent_workers=True)
    return TrainLoader, ValidLoader

if __name__ == '__main__':
    train_loader, test_loader = get_face_data()
    for x, y in train_loader:
        print(y[0])
        assert 0
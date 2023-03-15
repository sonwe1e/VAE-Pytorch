import torch
import torch.nn as nn

# label2onehot
def label2onehot(label, num_classes):
    batch_size = label.size(0)
    index = label.view(-1, 1)
    src = torch.ones(batch_size, 1).cuda(3)
    one_hot = torch.zeros(batch_size, num_classes).cuda(3)
    one_hot.scatter_(dim=1, index=index, src=src)
    return one_hot

if __name__ == '__main__':
    label = torch.arange(0, 10, device='cuda')
    onehot = label2onehot(label, 10)
    print(label)
    print(onehot)
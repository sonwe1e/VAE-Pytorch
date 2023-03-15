import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import cv2
from Model import CVAE
import torch.nn.functional as F
from Data import get_data
from utils import label2onehot
torch.set_float32_matmul_precision('high')

class PL(pl.LightningModule):
    def __init__(self):
        super(PL, self).__init__()
        self.model = CVAE()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, label2onehot(y, 10))
        x_hat, x, mu, logvar = output
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, label2onehot(y, 10))
        x_hat, x, mu, logvar = output
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        class_feature = torch.arange(10).reshape(10, 1).cuda()
        class_feature = label2onehot(class_feature, 10)
        test_z = torch.randn(10, 64, device='cuda')
        test_z = torch.cat([test_z, class_feature], dim=1)
        test_x = self.model.decoder(test_z)
        for i in range(10):
            cv2.imwrite(f'FromX/{i}.png', 255*x_hat[i].cpu().detach().numpy().reshape(28, 28))
        for i in range(10):
            cv2.imwrite(f'FromZ/{i}.png', 255*test_x[i].cpu().detach().numpy().reshape(28, 28))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4)

if __name__ == '__main__':
    train_loader, val_loader = get_data()
    model = PL()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='cvae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback], accelerator='gpu', devices=[4])
    trainer.fit(model, train_loader, val_loader)

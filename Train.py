import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import cv2
from MNISTModel import CVAE
from FaceModel import FaceVAE
import torch.nn.functional as F
from Data import get_digit_data, get_face_data
from utils import label2onehot
torch.set_float32_matmul_precision('high')

class PL(pl.LightningModule):
    def __init__(self):
        super(PL, self).__init__()
        self.model = FaceVAE()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, y)
        x_hat, x, mu, logvar = output
<<<<<<< HEAD
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.1 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
=======
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
>>>>>>> 157ad8c368c323894a7771af5ec784761f2f4e9e
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, y)
        x_hat, x, mu, logvar = output
<<<<<<< HEAD
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.1 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
=======
        loss = F.mse_loss(x_hat, x, reduction='sum') -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
>>>>>>> 157ad8c368c323894a7771af5ec784761f2f4e9e
        # class_feature = torch.arange(10).reshape(10, 1).cuda()
        # class_feature = label2onehot(class_feature, 10)
        class_feature = torch.sign(torch.randn((10, 40, 1, 1), device=self.device)-0.7)
        test_z = torch.randn((10, 120, 1, 1), device=self.device)
        test_z = torch.cat([test_z, class_feature], dim=1)
        test_z = self.model.decoder(test_z)
        for i in range(10):
            cv2.imwrite(f'FromX/{i}.png', 255*x_hat[i].permute(1, 2, 0).float().cpu().detach().numpy())
        for i in range(10):
            cv2.imwrite(f'FromZ/{i}.png', 255*test_z[i].permute(1, 2, 0).float().cpu().detach().numpy())
        self.log('val_loss', loss)
        self.log('mu', torch.mean(mu), prog_bar=True)
        self.log('logvar', torch.mean(logvar), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4)

if __name__ == '__main__':
    print('Getting data...')
    train_loader, val_loader = get_face_data()
    print('Data loaded.')
    model = PL()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='cvae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback],
                         accelerator='gpu', devices=[3])
    print('Start training...')
    trainer.fit(model, train_loader, val_loader)

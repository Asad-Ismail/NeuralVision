import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import pytorch_lightning as pl


class MoCo(pl.LightningModule):
    def __init__(self, backbone, dim=128, m=0.999, T=0.07):
        super().__init__()
        
        self.encoder_q = timm.create_model(backbone, pretrained=True, features_only=False, num_classes=dim)
        self.encoder_k = timm.create_model(backbone, pretrained=True, features_only=False, num_classes=dim)
        
        # Initialize the momentum encoder with the same weights as the query encoder
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        
        # Set the momentum encoder to not update its weights through backpropagation
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        self.dim = dim
        self.m = m
        self.T = T

        self.train_loss = pl.metrics.Mean('train_loss')

    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        k = self.encoder_k(x_k)
        return q, k

    def training_step(self, batch, batch_idx):
        x_q, x_k = batch
        q, k = self.forward(x_q, x_k)
        loss = self.contrastive_loss(q, k, self.m, self.T)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def contrastive_loss(self, q, k, m, T):
        batch_size = q.size(0)
        
        # Compute the similarity between query and keys
        sim_matrix = torch.matmul(q, k.t().detach())
        
        # Scale the similarity matrix with the temperature parameter
        sim_matrix /= T
        
        # Calculate the positive logits by taking the diagonal of the similarity matrix
        pos_logits = torch.diag(sim_matrix)
        
        # Calculate the negative logits from the similarity matrix (excluding diagonal)
        neg_logits = sim_matrix - torch.eye(batch_size).cuda() * 1e9
        
        # Compute the logits by concatenating the positive and negative logits
        logits = torch.cat((pos_logits.unsqueeze(1), neg_logits), dim=1)
        
        # Calculate the contrastive loss using the cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, torch.zeros(batch_size, dtype=torch.long).cuda())
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.encoder_q.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        x_q, x_k = batch
        q, k = self.forward(x_q, x_k)
        loss = self.contrastive_loss(q, k, self.m, self.T)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(outputs).mean()
        self.log('val_loss', val_loss, on_epoch=True)
        return {'val_loss': val_loss}


backbone = "resnet50"

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

train_dataset = datasets.STL10(root='./data', split='train+unlabeled', download=True, transform=train_transforms)
val_dataset = datasets.STL10(root='./data', split='test', download=True, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Define the Lightning trainer
trainer = pl.Trainer(gpus=1, max_epochs=200, progress_bar_refresh_rate=20,
                     checkpoint_callback=pl.callbacks.ModelCheckpoint(dirpath='./checkpoints',
                                                                      monitor='val_loss',
                                                                      mode='min',
                                                                      filename='best_model'),
                     logger=pl.loggers.TensorBoardLogger('logs/', name='MoCo'))

# Create the MoCo model and start training
moco_model = MoCo(backbone)
trainer.fit(moco_model, train_loader, val_loader)

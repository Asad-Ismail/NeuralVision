import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm


class MoCo(nn.Module):
    def __init__(self, backbone, dim=128, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        
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

    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        k = self.encoder_k(x_k)
        return q, k


def contrastive_loss(q, k, m, T):
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


backbone = "resnet50"
moco_model = MoCo(backbone).cuda()
optimizer = optim.SGD(moco_model.encoder_q.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)


num_epochs = 200

for epoch in range(num_epochs):
    for x_q, x_k in data_loader:
        x_q, x_k = x_q.cuda(), x_k.cuda()
        
        # Forward pass
        q, k = moco_model(x_q, x_k)
        
        # Compute the contrastive loss
        loss = contrastive_loss(q, k, moco_model.m, moco_model.T)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the momentum encoder with the moving average of the query encoder
        with torch.no_grad():
            for param_q, param_k in zip(moco_model.encoder_q.parameters(), moco_model.encoder_k.parameters()):
                param_k.data = param_k.data * moco_model.m + param_q.data * (1 - moco_model.m)

    # Print the loss for the current epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

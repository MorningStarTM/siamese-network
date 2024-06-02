import torch
import torch.nn as nn
import torch.optim as optim

class ContrastiveLoss(nn.Module):

    """
    This code for implement contrastive loss

    Args:  
        margin (float)

    Return:
        loss_contrastive 

    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    





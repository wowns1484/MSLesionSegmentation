import torch.nn as nn
from lib.losses3D.dice import DiceLoss
from lib.losses3D.basic import expand_as_one_hot
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
from torch.nn import functional as F

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss(classes=classes)
        self.classes=classes

    def forward(self, inputs, target, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(input)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = target.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

# class BCEDiceLoss(nn.Module):
#     """Linear combination of BCE and Dice losses3D"""

#     def __init__(self, alpha=1, beta=1, classes=4):
#         super(BCEDiceLoss, self).__init__()
#         self.alpha = alpha
#         self.bce = nn.BCEWithLogitsLoss()
#         self.beta = beta
#         self.dice = DiceLoss(classes=classes)
#         self.classes=classes

#     def forward(self, input, target):

#         target_expanded = expand_as_one_hot(target.long(), self.classes)

#         assert input.size() == target_expanded.size(), "'input' and 'target' must have the same shape"
        
#         loss_1 = self.alpha * self.bce(input, target_expanded)
#         loss_2, channel_score = self.beta * self.dice(input, target_expanded)
#         return  (loss_1+loss_2) , channel_score

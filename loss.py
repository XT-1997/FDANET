from __future__ import division
import torch
import torch.nn as nn

# 损失函数
class EuclideanLoss_with_Uncertainty(nn.Module):
    def __init__(self):
        super(EuclideanLoss_with_Uncertainty, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target, mask, certainty):
        loss_reg = self.pdist(pred, target)         # 对应的预测和真实坐标差值的二范数 [4 60 80]
        certainty_map = certainty
        certainty_map = torch.max(certainty.cuda(), torch.tensor(1e-6).cuda())  # certainty_map 置信度[4 60 80] [0,1]

        loss_map = 3 * torch.log(certainty_map) + loss_reg / (2 * certainty_map.pow(2))
        
        loss_map = loss_map * mask # 深度大于0的才有损失
        loss =torch.sum(loss_map) / mask.sum()

        if mask is not None:
            valid_pixel = mask.sum() + 1
            diff_coord_map = mask * loss_reg
        
        thres_coord_map = torch.max(diff_coord_map - 0.05, torch.tensor([0.]).cuda())
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        loss1 = torch.sum(loss_reg*mask) / mask.sum()
        return loss, accuracy, loss1

# coord+resnet18+PFFM+深度监督
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import sys
import os
from torch.autograd import Variable


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def conv_(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
    )

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers=(3, 4, 23, 3),
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=True,
                 output_stride=32,
                 ):

        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x32s = self.layer4(x_3)
        x = x32s

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        # xfc = self.fc(x)

        return x32s, x_3


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        print("loading resnet18 pretrained mdl.")
        model.load_state_dict(
            model_zoo.load_url(
                model_urls['resnet18'], model_dir='./'
            )
        )
    return model


def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        print("loading resnet34 pretrained mdl.")
        model.load_state_dict(
            model_zoo.load_url(
                model_urls['resnet34'], model_dir='./'
            )
        )
    return model


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = conv(in_chan, out_chan, stride=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class PyramidFeatureFusionMoudle(nn.Module):
    def __init__(self):
        super(PyramidFeatureFusionMoudle, self).__init__()
        self.ARM1 = AttentionRefinementModule(512, 512)
        self.ARM2 = AttentionRefinementModule(512, 512)
        self.cnn1 = conv(512, 512, stride=1)
        self.cnn2 = conv(512, 512, stride=1)

        # conv(2048 512)
        self.conv_fusion = conv(1536, 512, 1)
        self.conv_reg1 = conv(512, 256)
        self.conv_reg2 = conv(256, 128)
        self.conv_reg3 = conv(128, 128)
        self.conv_coord = conv_(128,3)
        self.conv_uncen = conv_(128,1)


    def forward(self,fusion_list):
        pffm_out = []
        pffm_out.append(fusion_list[0])
        ARM_out1 = self.ARM1(fusion_list[0])

        pffm2 = ARM_out1 + fusion_list[1]
        pffm2 = self.cnn1(pffm2)
        pffm_out.append(pffm2)

        ARM_out2 = self.ARM2(pffm2)
        pffm3 = ARM_out2 + fusion_list[2]
        pffm3 = self.cnn2(pffm3)
        pffm_out.append(pffm3)

        pffm_out = torch.cat(pffm_out,1)
        pffm_out = self.conv_fusion(pffm_out)
        
        #########位置编码
        #pffm_out = self.position_encdoing(pffm_out)

        pffm_out = self.conv_reg1(pffm_out)
        pffm_out = self.conv_reg2(pffm_out)
        pffm_out = self.conv_reg3(pffm_out)
        coordin = self.conv_coord(pffm_out)
        unconfi = torch.sigmoid(self.conv_uncen(pffm_out))
        #unconfi = self.conv_uncen(pffm_out)
        
        return coordin, unconfi

class FDANet(nn.Module):
    def __init__(self, training=True, dataset='7S'):
        super(FDANet, self).__init__()
        cnn = resnet18(pretrained=True)
        if dataset == '7S':
            self.cnn_pre = nn.Sequential(conv(5, 64), conv(64, 64, stride=2), cnn.maxpool)
        else:
            self.cnn_pre = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool)
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2; self.conv128_512 = conv(128,512,kernel_size=1)
        self.layer3 = cnn.layer3; self.conv256_512 = conv(256,512,kernel_size=1)
        self.layer4 = cnn.layer4

        
        # ------------------------------------深度监督部分
        self.supervise_conv_reg1 = conv(512, 256)
        self.supervise_conv_reg2 = conv(256, 128)
        self.supervise_conv_reg3 = conv(128, 128)
        self.supervise_conv_coord = conv_(128,3)
        self.supervise_conv_uncen = conv_(128,1)

        self.pffm = PyramidFeatureFusionMoudle()

        

    def forward(self, x):
        out = self.cnn_pre(x)

        fusion_out = []
        out = self.layer1(out);
        out = self.layer2(out); fusion_out.append(self.conv128_512(out))
        out = self.layer3(out); fusion_out.append(self.conv256_512(out))
        out = self.layer4(out); fusion_out.append(out)

        # ------------------------------------深度监督部分
        #deep_supervise = self.position(out)
        
        deep_supervise = self.supervise_conv_reg1(out)
        deep_supervise = self.supervise_conv_reg2(deep_supervise)
        deep_supervise = self.supervise_conv_reg3(deep_supervise)
        supervise_coord = self.supervise_conv_coord(deep_supervise)
        supervise_uncen = torch.sigmoid(self.supervise_conv_uncen(deep_supervise))
        # ------------------------------------深度监督部分
        fusion_out = list(reversed(fusion_out))

        coord_map, uncertainty_map = self.pffm(fusion_out)
        #print(coord_map.shape)
        return coord_map, uncertainty_map, supervise_coord, supervise_uncen





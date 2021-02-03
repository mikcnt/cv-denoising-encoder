import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as tf

from utils.layer import conv_layer
from utils.layer import maxpool
from utils.layer import transpose_conv
from utils.layer import conv_maxpool
from utils.layer import deconv_block


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoding block...
        self.conv_max1 = nn.Sequential(
            conv_layer(in_ch=3, out_ch=48, kernel=3),
            conv_maxpool(in_ch=48, out_ch=48, kernel=3),
        )
        self.conv_max2 = conv_maxpool(in_ch=48, out_ch=48, kernel=3)
        self.conv_max3 = conv_maxpool(in_ch=48, out_ch=48, kernel=3)
        self.conv_max4 = conv_maxpool(in_ch=48, out_ch=48, kernel=3)
        self.conv_max5 = nn.Sequential(
            conv_maxpool(in_ch=48, out_ch=48, kernel=3),
            conv_layer(in_ch=48, out_ch=48, kernel=3),
            transpose_conv(48, 48),
        )

        # Deconv blocks...
        self.dec_conv5 = deconv_block(in_ch=96, out_ch=96, kernel=3)
        self.dec_conv4 = deconv_block(in_ch=144, out_ch=96, kernel=3)
        self.dec_conv3 = deconv_block(in_ch=144, out_ch=96, kernel=3)
        self.dec_conv2 = deconv_block(in_ch=144, out_ch=96, kernel=3)
        self.dec_conv1 = nn.Sequential(
            conv_layer(in_ch=99, out_ch=64, kernel=3),
            conv_layer(in_ch=64, out_ch=32, kernel=3),
            conv_layer(in_ch=32, out_ch=3, kernel=3, activation=nn.Sigmoid()),
        )

    def forward(self, x):
        concats = [x]
        output = self.conv_max1(x)
        concats.append(output)
        output = self.conv_max2(output)
        concats.append(output)
        output = self.conv_max3(output)
        concats.append(output)
        output = self.conv_max4(output)
        concats.append(output)
        output = self.conv_max5(output)
        output = torch.cat(
            (output, concats.pop()), dim=1
        )  # concat output of pool4 on channel dimension
        output = self.dec_conv5(output)
        output = torch.cat(
            (output, concats.pop()), dim=1
        )  # concat output of pool3 on channel dimension
        output = self.dec_conv4(output)
        output = torch.cat(
            (output, concats.pop()), dim=1
        )  # concat output of pool2 on channel dimension
        output = self.dec_conv3(output)
        output = torch.cat(
            (output, concats.pop()), dim=1
        )  # concat output of pool1 on channel dimension
        output = self.dec_conv2(output)
        output = torch.cat((output, concats.pop()), dim=1)  # concat input
        output = self.dec_conv1(output)
        return output

from torch import nn
import torchvision.transforms as tf


def conv_layer(
    in_ch, out_ch, kernel, activation=nn.LeakyReLU(), stride=1, padding="same"
):
    """Convolutional block, composed by Conv2D, BatchNorm and non-linearity.

    Args:
        in_ch (int): Number of input channels for the convolution.
        out_ch (int): Number of output channels for the convolution.
        kernel (int): Filter size for the convolution.
        activation (nn.activation, optional): Non-linearity after the convolutional layer.
                                              Defaults to nn.LeakyReLU().
        stride (int, optional): Stride used in the convolutional layer. Defaults to 1.
        padding (str, optional): Zero-padding. If 'same', dimensions are kept.
                                 Defaults to "same".

    Returns:
        func: Convolutional block.
    """
    if padding == "same":
        padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        activation,
    )


def conv_maxpool(in_ch, out_ch, kernel, stride=1):
    return nn.Sequential(
        conv_layer(in_ch=in_ch, out_ch=out_ch, kernel=kernel, stride=stride), maxpool()
    )


def deconv_block(in_ch, out_ch, kernel, stride=1):
    return nn.Sequential(
        conv_layer(in_ch=in_ch, out_ch=out_ch, kernel=kernel, stride=stride),
        conv_layer(in_ch=out_ch, out_ch=out_ch, kernel=kernel, stride=stride),
        transpose_conv(96, 96),
    )


def maxpool(kernel=2):
    return nn.MaxPool2d(kernel_size=kernel)


def transpose_conv(in_ch, out_ch, kernel=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding)

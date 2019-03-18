from torch import nn as nn


def pool_2():
    """
    Returns the first pooling-layer in the modified VGG-16 architecture.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=64),
        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=64),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=128),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=128),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def pool_3():
    """
    Returns the second pooling-layer in the modified VGG-16 architecture.
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def conv_4():
    """
    Returns the first convolutional layer in the modified VGG-16 architecture
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU()
    )


def conv_5():
    """
    Returns the second convolutional layer in the modified VGG-16 architecture
    """
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=512),
        nn.ReLU()
    )


def skip_pool_2():
    """
    Returns the skip-connections corresponding to the first pooling layer.
    """
    return nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.GroupNorm(num_groups=32, num_channels=128)
    )


def skip_pool_3():
    """
    Returns the skip-connections corresponding to the second pooling layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.GroupNorm(num_groups=32, num_channels=128)
    )


def skip_conv_4():
    """
    Returns the skip-connections corresponding to the first convolutional layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.GroupNorm(num_groups=32, num_channels=128)
    )


def skip_conv_5():
    """
    Returns the skip-connections corresponding to the second convolutional layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.GroupNorm(num_groups=32, num_channels=128),
        nn.Upsample(scale_factor=2, mode='bilinear')
    )


def fuse_features():
    """
    Returns the final stack of image features.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.GroupNorm(num_groups=32, num_channels=128)
    )
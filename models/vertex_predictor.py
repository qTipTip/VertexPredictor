import torch
import torch.nn as nn

from models.aux_models import pool_2, pool_3, conv_4, conv_5, skip_pool_2, skip_pool_3, skip_conv_4, skip_conv_5, \
    fuse_features


class ImageEncoder(nn.Module):
    """
    This class extracts image features from an image of size 3x224x224, outputting a
    stack of feature maps of shape 128xDxD where D is the desired output resolution.
    It is based on a modified VGG-16 architecture with skip-connections to a fused
    set of feature maps from varying network depths.
    """

    def __init__(self, output_resolution=28):
        super().__init__()

        self.output_resolution = output_resolution

        # network layers
        self.pool_2 = pool_2()
        self.pool_3 = pool_3()
        self.conv_4 = conv_4()
        self.conv_5 = conv_5()

        # skip connections with relevant transformations
        self.skip_pool_2 = skip_pool_2()
        self.skip_pool_3 = skip_pool_3()
        self.skip_conv_4 = skip_conv_4()
        self.skip_conv_5 = skip_conv_5()

        # fused stack of features from each skip connection
        self.fused_features = fuse_features()

    def forward(self, x):
        """
        Compute one forward pass in the network.
        :param x: input batch of size (B, 3, 128, 128)
        :return: image features of size (B, 128, 28, 28)
        """

        pool_2_out = self.pool_2(x)
        pool_3_out = self.pool_3(pool_2_out)
        conv_4_out = self.conv_4(pool_3_out)
        conv_5_out = self.conv_5(conv_4_out)

        pool_2_out = self.skip_pool_2(pool_2_out)
        pool_3_out = self.skip_pool_3(pool_3_out)
        conv_4_out = self.skip_conv_4(conv_4_out)
        conv_5_out = self.skip_conv_5(conv_5_out)

        fused = torch.cat((pool_2_out, pool_3_out, conv_4_out, conv_5_out))
        fused = self.fused_features(fused)

        return fused

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        :param x: input batch of size (B, 3, 224, 224)
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


def boundary_layer():
    return nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=1),
        nn.GroupNorm(num_groups=32, num_channels=128),
        nn.ReLU(),
    )


def vertex_layer():
    return nn.Sequential(
        nn.Linear(in_features=784 + 28 * 28 * 32, out_features=784),
        nn.Sigmoid()
    )


class VertexPredictor(nn.Module):
    """
    This class uses the extracted image features from the ImageEncoder-class, and predicts
    a set of vertices, their connectivity (edges), and which vertex is the first in the polygon.
    """

    def __init__(self, output_resolution=28):
        super().__init__()

        self.output_resolution = output_resolution
        self.image_encoder = ImageEncoder(output_resolution=output_resolution)
        self.boundary_layer = boundary_layer()
        self.boundary_fc = nn.Linear(32 * 28 * 28, out_features=784)
        self.vertex_layer = vertex_layer()

    def forward(self, x):
        """
        Compute one forward pass in the network.
        :param x: input batch of size (B, 3, 224, 224)
        :return:
        """

        image_features = self.image_encoder(x)
        boundary_features = self.boundary_layer(image_features)
        boundary_features = boundary_features.view(boundary_features.shape[0], -1)
        boundary_features = F.sigmoid(self.boundary_fc(boundary_features))

        boundary_and_image = torch.cat((image_features, boundary_features))
        vertex_features = F.sigmoid(self.vertex_layer(boundary_and_image))

        print(vertex_features)

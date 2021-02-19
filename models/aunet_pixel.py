import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_leru(in_channels, out_channels,
        kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=5, padding=2
        )
        self.conv7 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=7, padding=3
        )

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.sigmoid(self.conv1(x))
        self.layer3 = F.sigmoid(self.conv3(x))
        self.layer5 = F.sigmoid(self.conv5(x))
        self.layer7 = F.sigmoid(self.conv7(x))

        out = torch.cat([self.layer1, self.layer3, self.layer5, self.layer7, x], 1)

        return out


class PAUnet(nn.Module):
    def __init__(self, input_channels, nclasses):
        super().__init__()

        _filters = [64, 128, 256, 512, 1024]
        _filters = [32, 64, 128, 256, 512]
        filters = [16, 32, 64, 128, 256]

        self.att = AttentionBlock(filters[0])

        # go down
        self.conv1 = conv_bn_leru(input_channels,filters[0])
        self.conv2 = conv_bn_leru(filters[0], filters[1])
        self.conv3 = conv_bn_leru(filters[1], filters[2])
        self.conv4 = conv_bn_leru(filters[2], filters[3])
        self.conv5 = conv_bn_leru(filters[3], filters[4])
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(filters[4], filters[3])
        self.conv6 = conv_bn_leru(filters[4], filters[3])
        self.up_pool7 = up_pooling(filters[3], filters[2])
        self.conv7 = conv_bn_leru(filters[3], filters[2])
        self.up_pool8 = up_pooling(filters[2], filters[1])
        self.conv8 = conv_bn_leru(filters[2], filters[1])
        self.up_pool9 = up_pooling(filters[1], filters[0])
        self.conv9 = conv_bn_leru(filters[1], filters[0])

        self.conv10 = nn.Conv2d(filters[0]+4, nclasses, 1)


        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0,
                                       mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # attention
        # x5a = self.att(x5)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        x9a = self.att(x9)


        output = self.conv10(x9a)
        # output = torch.softmax(output, dim=1)
        output = torch.sigmoid(output)

        return output

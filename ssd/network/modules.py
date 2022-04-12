import torch
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):
    """
    正規化レイヤー
    """
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


class DeconvModule(nn.Module):
    """
    DssDモジュール
    """
    def __init__(self, in_channel, deconv_kernel_size):
        super(DeconvModule, self).__init__()

        # 順伝播モジュール
        self.deconv = nn.ConvTranspose2d(512, 512, kernel_size=deconv_kernel_size, stride=2)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        # 合流モジュール
        self.conv2 = nn.Conv2d(in_channel, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):

        # deconv
        x1 = self.deconv(x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)

        # 合流層
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)

        # 合流
        out = x1 + x2
        out = self.relu(out)

        return out


class PredModule(nn.Module):
    """
    出力モジュール
    """
    def __init__(self, bbox_num):
        super(PredModule, self).__init__()

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 1024, kernel_size=1)

        self.conv_bypath = nn.Conv2d(512, 1024, kernel_size=1)

        self.conv_loc = nn.Conv2d(1024, bbox_num * 4, kernel_size=3, padding=1)
        self.conv_conf = nn.Conv2d(1024, bbox_num, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += self.conv_bypath(identity)

        loc = self.conv_loc(x)
        conf = self.conv_conf(x)

        return loc, conf

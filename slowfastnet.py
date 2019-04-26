import torch.nn as nn
import torch


def conv1x1x1(inplanes, planes, stride=1):
    return nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)


def conv1x3x3(inplanes, planes, stride=1, padding=(0, 1, 1)):
    return nn.Conv3d(inplanes, planes, (1, 3, 3), stride=stride, padding=padding, bias=False)


def conv3x1x1(inplanes, planes, stride=1, padding=(1, 0, 0)):
    return nn.Conv3d(inplanes, planes, (3, 1, 1), stride=stride, padding=padding, bias=False)


class Degenerate_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Degenerate_Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Non_degenerate_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Non_degenerate_Bottleneck, self).__init__()
        self.conv1 = conv3x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class slowfast(nn.Module):

    def __init__(self, layers, num_classes=174):
        super(slowfast, self).__init__()

        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res2 = self._make_layer(Non_degenerate_Bottleneck, 8, 8, layers[0])
        self.fast_res3 = self._make_layer(Non_degenerate_Bottleneck, 32, 16, layers[1], stride=(1, 2, 2))
        self.fast_res4 = self._make_layer(Non_degenerate_Bottleneck, 64, 32, layers[2], stride=(1, 2, 2))
        self.fast_res5 = self._make_layer(Non_degenerate_Bottleneck, 128, 64, layers[3], stride=(1, 2, 2))

        self.slow_res2 = self._make_layer(Degenerate_Bottleneck, 64 + 8 * 2, 64, layers[0])
        self.slow_res3 = self._make_layer(Degenerate_Bottleneck, 256 + 32 * 2, 128, layers[1], stride=(1, 2, 2))
        self.slow_res4 = self._make_layer(Non_degenerate_Bottleneck, 512 + 64 * 2, 256, layers[2], stride=(1, 2, 2))
        self.slow_res5 = self._make_layer(Non_degenerate_Bottleneck, 1024 + 128 * 2, 512, layers[3], stride=(1, 2, 2))

        self.tconv1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256 + 2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # the input in the paper should be [N,C,T*τ,H,W]
        # the fast way's frame interval is 2 and the low way 16
        # so we only input [N,C,T/2*τ,H,W] to save memory and time.

        fast_out, lateral = self._fast_net(x)
        slow_out = self._slow_net(x[:, :, ::8, ...], lateral)
        fusion_out = torch.cat([fast_out, slow_out], dim=1)
        output = self.fc(fusion_out)

        return output

    def _slow_net(self, x, lateral):
        out = self.slow_conv1(x)
        out = self.slow_bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = torch.cat([out, lateral[0]], dim=1)

        out = self.slow_res2(out)
        out = torch.cat([out, lateral[1]], dim=1)

        out = self.slow_res3(out)
        out = torch.cat([out, lateral[2]], dim=1)

        out = self.slow_res4(out)
        out = torch.cat([out, lateral[3]], dim=1)

        out = self.slow_res5(out)

        slow_out = self.avgpool(out).view(x.size(0), -1)

        return slow_out

    def _fast_net(self, x):
        lateral = []
        out = self.fast_conv1(x)
        out = self.fast_bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        lateral.append(self.tconv1(out))

        out = self.fast_res2(out)
        lateral.append(self.tconv2(out))
        out = self.fast_res3(out)
        lateral.append(self.tconv3(out))
        out = self.fast_res4(out)
        lateral.append(self.tconv4(out))
        out = self.fast_res5(out)

        fast_out = self.avgpool(out).view(x.size(0), -1)

        return fast_out, lateral


def SlowFastNet(num_classes=174):
    model = slowfast([3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = SlowFastNet()
    from torchsummary import summary
    summary(model, (3, 48, 224, 224))

    from tools import print_model_parm_flops
    print_model_parm_flops(model,torch.randn(1,3,48,224,224))

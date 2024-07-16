import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), stride=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(dilation, dilation),
                              dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class AtrousConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), atrous_rates=[4, 8, 16]):
        super(AtrousConvBlock, self).__init__()
        self.atrous_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(rate, rate), dilation=rate) for rate
             in atrous_rates])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1)

    def forward(self, x):
        atrous_outs = [conv(x) for conv in self.atrous_convs]
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        atrous_outs.append(global_feat)
        x = torch.cat(atrous_outs, dim=1)
        return self.conv1x1(x)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        self.atrous_conv_block = AtrousConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.atrous_conv_block(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(20000, 128)
        self.conv_block = ConvBlock(128, 256, stride=1)
        self.atrous_conv_block = ConvBlock(256, 256, stride=1, dilation=2)
        self.aspp_module = ASPPModule(256, 256)
        self.shortcut_layer = nn.Conv2d(256, 64, kernel_size=1)
        self.output_layer = nn.Conv2d(64, 9, kernel_size=1)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.permute(0, 3, 1, 2)  # Assuming input shape [batch_size, seq_len, embedding_dim]
        x = self.conv_block(x)
        x = self.atrous_conv_block(x)
        x = self.aspp_module(x)
        x = self.shortcut_layer(x)
        x = self.output_layer(x)
        return x


# Example usage:
model = Model()
input_data = torch.randint(0, 20000, (10,2))
output = model(input_data)
print(output.shape)  # Should print [batch_size, 9, height, width]

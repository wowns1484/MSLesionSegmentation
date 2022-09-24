import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self, num_conva, in_channels, growthRate = 48, **kwargs) -> None:
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()

        for idx in range(num_conva):
            self.net.add_module(name=f'Dense + {idx}', module=self.doubleConvBlock(in_channels + idx*growthRate, growthRate))
    
    def doubleConvBlock(self, in_channels, out_channels):
        inter_planes = out_channels * 4
        block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, inter_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(inter_planes),
            nn.Conv3d(inter_planes, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        return block
        
    def forward(self, x):
        for block in self.net:
            out = block(x)
            x = torch.concat([x, out], dim=1)

        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.droprate = dropRate
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        out = self.avg_pool(out)
        return out

in_channels_dense = [64, 128, 256, 512]
num_blocks = [6, 12, 36, 24]
grows_rate = 48

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropRate = 0.0):
        #input dimsnsion을 정하고, output dimension을 정하고(growh_rate임), dropRate를 정함.
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace = True) # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.droprate > 0:
            out = F.dropout (out, p = self.droprate, training = self.training)
        
        return out, self.max_pool(out)

class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(TransposeBlock, self).__init__()
        self.convTrans = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convTrans(x)

class DenseUNet3D(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features_encoder=[32, 64, 128, 256], features_decoder=[1408, 512, 256, 128], features_double=[2368, 896, 448, 96]
    ):
        super(DenseUNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs_dense = nn.ModuleList()
        self.downs_trans = nn.ModuleList()
        self.basic_block = BasicBlock(1, 32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

        # Down part of UNET
        for idx, (feature, num_block) in enumerate(zip(features_encoder, num_blocks)):
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs_dense.append(DenseBlock(num_block, feature, grows_rate))
            in_channels = feature

            if idx < 3:
                self.downs_trans.append(TransitionBlock(in_channels + num_block*grows_rate, features_encoder[idx+1]))

        # Up part of UNET
        for idx, (feature, double, encoder) in enumerate(zip(features_decoder, features_double, reversed(features_encoder))):
            self.ups.append(TransposeBlock(feature, encoder*2))
            self.ups.append(DoubleConv(double, encoder*2))

        self.final_convTrans = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.basic_block(x)
        skip_connections.append(x[0])
        x = x[1]

        for down, down_dense, down_trans in zip(self.downs, self.downs_dense, self.downs_trans):
            x = down_dense(x)
            # print(f"x shape from down sample block: {x.shape}")
            skip_connections.append(x)
            x = down_trans(x)
            # print(f"x shape from down sample block after pool: {x.shape}")

        x = self.downs_dense[-1](x)
        # print(x.shape)
        skip_connections = skip_connections[::-1]
        # print(f"skip connections list length: {len(skip_connections)}")

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # print(f"x shape from up sample block before concat_skip: {x.shape}")

            if x.shape[2:] != skip_connection.shape[2:]:
                # print(f"x shape different to skip shape: {x.shape, skip_connection.shape}")
                p3d = (0, 1, 0, 0, 0, 1)
                x = F.pad(x, p3d)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            # print(f"x shape from up sample block after concate skip & x: {concat_skip.shape}")
            x = self.ups[idx+1](concat_skip)
            # print(f"x shape from up sample block after concate complete: {x.shape}")
        
        x = self.final_convTrans(x)
        # print(f"final convTranspose x shape: {x.shape}")
        x = self.final_conv(x)
        # print(f"final conv x shape: {x.shape}")
        x = self.sigmoid(x)

        return x
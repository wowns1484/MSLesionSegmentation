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
        ).to('cuda')

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
num_blocks = [5, 10, 30, 20]
grows_rate = 48

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropRate = 0.0):
        #input dimsnsion을 정하고, output dimension을 정하고(growh_rate임), dropRate를 정함.
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace = True) # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1).to('cuda')
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

class unetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class NestedDenseUNet3D(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features_encoder=[32, 64, 128, 256], features_decoder=[1216, 512, 256, 128], features_double=[1824, 800, 464, 154]
    ):
        super(NestedDenseUNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs_dense = nn.ModuleList()
        self.downs_trans = nn.ModuleList()
        self.basic_block = BasicBlock(1, 32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2)

        # Down part of UNET
        for idx, (feature, num_block) in enumerate(zip(features_encoder, num_blocks)):
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs_dense.append(DenseBlock(num_block, feature, grows_rate))
            in_channels = feature

            if idx < 3:
                self.downs_trans.append(TransitionBlock(in_channels + num_block*grows_rate, features_encoder[idx+1]))

        # Up part of UNET
        for idx, (feature, double, encoder) in enumerate(zip(features_decoder, features_double, reversed(features_encoder))):
            self.ups.append(TransposeBlock(feature, encoder))
            self.ups.append(DoubleConv(double, encoder*2))

        self.final_convTrans = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.basic_block(x)
        x0_0 = x[0]
        x1_0 = x[1]
        x1_0_dense = self.downs_dense[0](x1_0)
        x0_1 = DoubleConv(304, 30)
        x0_1 = x0_1(torch.cat([x0_0, self.up(x1_0_dense)], 1))
        x1_0 = self.downs_trans[0](x1_0_dense)

        x2_0_dense = self.downs_dense[1](x1_0)
        x1_1 = DoubleConv(816, 64)
        x2_0_trans = TransposeBlock(x2_0_dense.shape[1], x2_0_dense.shape[1]).to('cuda')
        x1_1 = x1_1(torch.cat([x1_0_dense, x2_0_trans(x2_0_dense)], 1))
        x0_2 = DoubleConv(126, 30)
        x1_1_trans = TransposeBlock(x1_1.shape[1], x1_1.shape[1]).to('cuda')
        x0_2 = x0_2(torch.cat([x0_0, x0_1, x1_1_trans(x1_1)], 1))
        x2_0 = self.downs_trans[1](x2_0_dense)

        x3_0_dense = self.downs_dense[2](x2_0)
        x2_1 = DoubleConv(2112, 128)
        x3_0_trans = TransposeBlock(x3_0_dense.shape[1], x3_0_dense.shape[1]).to('cuda')
        x2_1 = x2_1(torch.cat([x2_0_dense, x3_0_trans(x3_0_dense)], 1))
        x1_2 = DoubleConv(464, 64)
        x2_1_trans = TransposeBlock(x2_1.shape[1], x2_1.shape[1]).to('cuda')
        x1_2 = x1_2(torch.cat([x1_0_dense, x1_1, x2_1_trans(x2_1)], 1))
        x0_3 = DoubleConv(156, 30)
        x1_2_trans = TransposeBlock(x1_2.shape[1], x1_2.shape[1]).to('cuda')
        x0_3 = x0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_trans(x1_2)], 1))
        x3_0 = self.downs_trans[2](x3_0_dense)

        x4_0 = self.downs_dense[3](x3_0)
        p3d = (0, 1, 0, 0, 0, 1)
        x4_0_up = self.ups[0](x4_0)
        x4_0_up = F.pad(x4_0_up, p3d)
        x3_1 = self.ups[1](torch.cat([x3_0_dense, x4_0_up], 1))
        x3_1_up = self.ups[2](x3_1)
        x2_2 = self.ups[3](torch.cat([x2_0_dense, x2_1, x3_1_up], 1))
        x2_2_up = self.ups[4](x2_2)
        x1_3 = self.ups[5](torch.cat([x1_0_dense, x1_1, x1_2, x2_2_up], 1))
        x1_3_up = self.ups[6](x1_3)
        x0_4 = self.ups[7](torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up], 1))

        ouput = self.final_convTrans(x0_4)
        ouput = self.final_conv(ouput)
        ouput = self.sigmoid(ouput)

        return ouput


# class NestedDenseUNet3D(nn.Module):
#     def __init__(
#             self, in_channels=3, out_channels=1, features_encoder=[16, 32, 64, 128], features_decoder=[1280, 256, 128, 64], features_double=[2048, 864, 496, 144]
#     ):
#         super(NestedDenseUNet3D, self).__init__()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.downs_dense = nn.ModuleList()
#         self.downs_trans = nn.ModuleList()
#         self.basic_block = BasicBlock(1, 16)
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.sigmoid = nn.Sigmoid()
#         self.up = nn.Upsample(scale_factor=2)

#         # Down part of UNET
#         for idx, (feature, num_block) in enumerate(zip(features_encoder, num_blocks)):
#             self.downs.append(DoubleConv(in_channels, feature))
#             self.downs_dense.append(DenseBlock(num_block, feature, grows_rate))
#             in_channels = feature

#             if idx < 3:
#                 self.downs_trans.append(TransitionBlock(in_channels + num_block*grows_rate, features_encoder[idx+1]))

#         # Up part of UNET
#         for idx, (feature, double, encoder) in enumerate(zip(features_decoder, features_double, reversed(features_encoder))):
#             self.ups.append(TransposeBlock(feature, encoder*2))
#             self.ups.append(DoubleConv(double, encoder*2))

#         self.final_convTrans = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
#         self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.basic_block(x)
#         x0_0 = x[0]
#         x1_0 = x[1]
#         x1_0_dense = self.downs_dense[0](x1_0)
#         x0_1 = DoubleConv(320, 32)
#         x0_1 = x0_1(torch.cat([x0_0, self.up(x1_0_dense)], 1))
#         x1_0 = self.downs_trans[0](x1_0_dense)

#         x2_0_dense = self.downs_dense[1](x1_0)
#         x1_1 = DoubleConv(912, 64)
#         x2_0_trans = TransposeBlock(x2_0_dense.shape[1], x2_0_dense.shape[1]).to('cuda')
#         x1_1 = x1_1(torch.cat([x1_0_dense, x2_0_trans(x2_0_dense)], 1))
#         x0_2 = DoubleConv(112, 32)
#         x1_1_trans = TransposeBlock(x1_1.shape[1], x1_1.shape[1]).to('cuda')
#         x0_2 = x0_2(torch.cat([x0_0, x0_1, x1_1_trans(x1_1)], 1))
#         x2_0 = self.downs_trans[1](x2_0_dense)

#         x3_0_dense = self.downs_dense[2](x2_0)
#         x2_1 = DoubleConv(2400, 128)
#         x3_0_trans = TransposeBlock(x3_0_dense.shape[1], x3_0_dense.shape[1]).to('cuda')
#         x2_1 = x2_1(torch.cat([x2_0_dense, x3_0_trans(x3_0_dense)], 1))
#         x1_2 = DoubleConv(496, 64)
#         x2_1_trans = TransposeBlock(x2_1.shape[1], x2_1.shape[1]).to('cuda')
#         x1_2 = x1_2(torch.cat([x1_0_dense, x1_1, x2_1_trans(x2_1)], 1))
#         x0_3 = DoubleConv(144, 32)
#         x1_2_trans = TransposeBlock(x1_2.shape[1], x1_2.shape[1]).to('cuda')
#         x0_3 = x0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_trans(x1_2)], 1))
#         x3_0 = self.downs_trans[2](x3_0_dense)

#         x4_0 = self.downs_dense[3](x3_0)
#         p3d = (0, 1, 0, 0, 0, 1)
#         x4_0_up = self.ups[0](x4_0)
#         x4_0_up = F.pad(x4_0_up, p3d)
#         x3_1 = self.ups[1](torch.cat([x3_0_dense, x4_0_up], 1))
#         x3_1_up = self.ups[2](x3_1)
#         x2_2 = self.ups[3](torch.cat([x2_0_dense, x2_1, x3_1_up], 1))
#         x2_2_up = self.ups[4](x2_2)
#         x1_3 = self.ups[5](torch.cat([x1_0_dense, x1_1, x1_2, x2_2_up], 1))
#         x1_3_up = self.ups[6](x1_3)
#         x0_4 = self.ups[7](torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up], 1))

#         ouput = self.final_convTrans(x0_4)
#         ouput = self.final_conv(ouput)
#         ouput = self.sigmoid(ouput)

#         return ouput
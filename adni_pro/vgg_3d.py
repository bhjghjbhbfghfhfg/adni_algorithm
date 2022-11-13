from torch import nn
import torch
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=8,kernel_size=3),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.Conv3d(in_channels=8,out_channels=8,kernel_size=3),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv3d(in_channels=8,out_channels=16,kernel_size=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(in_channels=16,out_channels=16,kernel_size=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(2,2)
        )

        self.layer3=nn.Sequential(
            nn.Conv3d(in_channels=16,out_channels=32,kernel_size=3),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.MaxPool3d(2,2)
        )

        self.layer4=nn.Sequential(
            nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.MaxPool3d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))

        # self.fc=nn.Sequential(
        #     nn.Linear(128,128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #
        #     nn.Linear(128,64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #
        #     nn.Linear(64,2)
        # )


    def forward(self,x,epoch=None):
        x=self.layer1(x)
        # import pdb;pdb.set_trace()
        x=self.layer2(x)
        x=self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(-1, 128)
        # x=self.fc(x)
        return x


# VGG = MyNet().cuda()
# x = torch.randn(1, 1, 110, 110, 110).cuda()
# y = VGG(x)
# print(y)
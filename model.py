import torch
import torch.nn as nn
from spectral import SpectralNorm
import numpy as np


class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x)
            .view(m_batchsize, -1, width * height)
            .permute(0, 2, 1)
        )  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 8
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.LeakyReLU(0.1),
        )
        curr_dim = conv_dim * mult
        self.layer2 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)
            ),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = int(curr_dim / 2)
        self.layer3 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)
            ),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = int(curr_dim / 2)
        self.layer4 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)
            ),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = int(curr_dim / 2)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1), nn.Tanh()
        )
        self.attn1 = Self_Attn(128, "relu")
        self.attn2 = Self_Attn(64, "relu")

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)

        return out, p1, p2


class Discriminator(nn.Module):
    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)), nn.LeakyReLU(0.1)
        )
        curr_dim = conv_dim
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = curr_dim * 2
        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = curr_dim * 2
        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )
        curr_dim = curr_dim * 2
        self.layer5 = nn.Sequential(nn.Conv2d(curr_dim, 1, 4))
        self.attn1 = Self_Attn(256, "relu")
        self.attn2 = Self_Attn(512, "relu")

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)

        return out.squeeze(), p1, p2

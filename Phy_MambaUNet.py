import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from .cbam import *


class PhysAdvectionDiffusion(nn.Module):
    """Approximate '∂x/∂t = -u·∇x + κΔx' """
    def __init__(self, channels, kappa=0.1):  
        super().__init__()
        self.channels = channels
        # 3×3 Laplacian kernel (learnable diffusion coefficient)
        laplace = torch.tensor([[0, 1, 0],
                                [1,-4, 1],
                                [0, 1, 0]], dtype=torch.float32) / 4.0    
        self.register_buffer("laplace", laplace[None,None])  
        self.kappa = nn.Parameter(torch.full((1,), kappa))  # learnable diffusion coefficient
        # depthwise conv used for advection (shift); initialized to zero → no advection at the beginning
        # construct two depthwise convolution layers: advection part
        self.adv_dx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.adv_dy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):  
        # Laplacian diffusion
        diff = F.conv2d(x, self.laplace.repeat(self.channels,1,1,1), padding=1, groups=self.channels)
        # advection (approximate ∂x/∂x, ∂x/∂y using learnable shift kernels)
        adv = self.adv_dx(x) + self.adv_dy(x)
        return x + self.kappa * diff - adv     # simple Euler step

class PMF_Block(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        
        # -------- Branch 1: Physics --------
        self.physics = PhysAdvectionDiffusion(input_dim)

        # -------- Branch 2: Mamba --------
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(d_model=input_dim, d_state=d_state,
                           d_conv=d_conv, expand=expand)

        # learnable α, β, initialized as β=1, α=0 to ensure equivalence with the original model at the beginning
        self.alpha  = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.beta   = nn.Parameter(torch.ones (1, dtype=torch.float32))

        self.fusion_proj = nn.Conv2d(input_dim, output_dim, 1, bias=False)

    def forward(self, x):
        if x.dtype == torch.float16:  # AMP
            x = x.float()
        B, C, H, W = x.shape

        # ---- Physics branch (keep 4-D) ----
        phys_out = self.physics(x)           # (B,C,H,W)

        # ---- Mamba branch (flatten → LN → Mamba) ----
        x_flat  = x.reshape(B, C, H*W).transpose(-1, -2)  # (B, HW, C)
        m_out   = self.mamba(self.norm(x_flat))
        m_out   = m_out.transpose(-1, -2).reshape(B, C, H, W)  # (B,C,H,W)

        # ---- weighted fusion ----
        fused = self.alpha * phys_out + self.beta * m_out    # (B,C,H,W)
        out   = self.fusion_proj(fused)                      # (B,output_dim,H,W)
        return out



class Phys_MambaUNet(nn.Module):
    def __init__(self, predicted_frames=3, input_frames=5, c_list=[8, 16, 24, 32, 48, 64], split_att='fc', bridge=True):
        super().__init__()

        self.encoder1 = nn.Sequential(nn.Conv2d(input_frames, c_list[0], 3, stride=1, padding=1))
        self.CBAM1 = CBAM(c_list[0])
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.CBAM2 = CBAM(c_list[1])
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.CBAM3 = CBAM(c_list[2])
        self.encoder4 = nn.Sequential(PMF_Block(input_dim=c_list[2], output_dim=c_list[3]))
        self.CBAM4 = CBAM(c_list[3])
        self.encoder5 = nn.Sequential(PMF_Block(input_dim=c_list[3], output_dim=c_list[4]))
        self.CBAM5 = CBAM(c_list[4])
        self.encoder6 = nn.Sequential(PMF_Block(input_dim=c_list[4], output_dim=c_list[5]))

        self.decoder1 = nn.Sequential(PMF_Block(input_dim=c_list[5], output_dim=c_list[4]))
        self.decoder2 = nn.Sequential(PMF_Block(input_dim=c_list[4], output_dim=c_list[3]))
        self.decoder3 = nn.Sequential(PMF_Block(input_dim=c_list[3], output_dim=c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))

        self.contr1 = nn.Sequential(nn.ConvTranspose2d(c_list[3], c_list[3], 2, stride=2))
        self.contr2 = nn.Sequential(nn.ConvTranspose2d(c_list[2], c_list[2], 2, stride=2))
        self.contr3 = nn.Sequential(nn.ConvTranspose2d(c_list[1], c_list[1], 2, stride=2))
        self.contr4 = nn.Sequential(nn.ConvTranspose2d(c_list[0], c_list[0], 2, stride=2))
        self.contr5 = nn.Sequential(nn.ConvTranspose2d(c_list[0], c_list[0], 2, stride=2))

        self.fusion_conv5 = nn.Conv2d(c_list[4]*2, c_list[4], kernel_size=1)
        self.fusion_conv4 = nn.Conv2d(c_list[3]*2, c_list[3], kernel_size=1)
        self.fusion_conv3 = nn.Conv2d(c_list[2]*2, c_list[2], kernel_size=1)
        self.fusion_conv2 = nn.Conv2d(c_list[1]*2, c_list[1], kernel_size=1)
        self.fusion_conv1 = nn.Conv2d(c_list[0]*2, c_list[0], kernel_size=1)

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.dbn6 = nn.GroupNorm(4, c_list[0])
        self.dbn7 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], c_list[0], kernel_size=1)

        self.S1 = nn.Conv2d(c_list[0], predicted_frames, 3, 1, 1)
        self.S = nn.Conv2d(predicted_frames, predicted_frames, 3, 1, 1)
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = self.CBAM1(out)
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = self.CBAM2(out)
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = self.CBAM3(out)
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = self.CBAM4(out)
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = self.CBAM5(out)

        out = F.gelu(self.ebn6(self.encoder6(out)))
        
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.cat([out5, t5], dim=1)
        out5 = self.fusion_conv5(out5)

        out4 = F.gelu(self.contr1(self.dbn2(self.decoder2(out5))))
        out4 = torch.cat([out4, t4], dim=1)
        out4 = self.fusion_conv4(out4)

        out3 = F.gelu(self.contr2(self.dbn3(self.decoder3(out4))))
        out3 = torch.cat([out3, t3], dim=1)
        out3 = self.fusion_conv3(out3)

        out2 = F.gelu(self.contr3(self.dbn4(self.decoder4(out3))))
        out2 = torch.cat([out2, t2], dim=1)
        out2 = self.fusion_conv2(out2)

        out1 = F.gelu(self.contr4(self.dbn5(self.decoder5(out2))))
        out1 = torch.cat([out1, t1], dim=1)
        out1 = self.fusion_conv1(out1)

        out0 = F.gelu(self.contr5(self.dbn6(self.final(out1))))
        out0 = self.S1(out0)
        out0 = out0 + x[:, -1, ...].unsqueeze(1)
        out0 = self.S(out0)
        out00 = out0 * torch.sigmoid(self.beta * out0)
        return out00
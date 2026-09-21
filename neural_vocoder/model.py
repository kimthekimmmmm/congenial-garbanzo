from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

@dataclass
class VocoderConfig:
    sample_rate: int = 48000
    hop_size: int = 300
    n_mels: int = 100
    channels: int = 256
    upsample_rates: tuple[int, ...] = (5, 5, 3, 4)
    upsample_kernels: tuple[int, ...] = (10, 10, 6, 8)
    resblock_kernels: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    speaker_dim: int = 128
    style_dim: int = 128
    def __post_init__(self):
        if math.prod(self.upsample_rates) != self.hop_size:
            raise ValueError("Product of upsample_rates must equal hop_size for sample-accurate alignment")

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: Sequence[int]):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=(kernel*d-d)//2)) for d in dilations])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel, 1, padding=(kernel-1)//2)) for _ in dilations])
    def forward(self, x: Tensor) -> Tensor:
        for a, b in zip(self.convs1, self.convs2):
            y = b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
            x = x + y
        return x

class HarmonicSource(nn.Module):
    """Differentiable voiced sinusoidal source plus controlled breath/noise."""
    def __init__(self, sample_rate: int, harmonics: int = 8):
        super().__init__(); self.sample_rate = sample_rate; self.harmonics = harmonics
        self.harmonic_gain = nn.Parameter(torch.ones(harmonics) / harmonics)
    def forward(self, f0: Tensor, voiced: Tensor, breath: Tensor | None, samples: int) -> Tensor:
        # Inputs are [B, frames]; interpolating F0 before phase integration prevents stair-step vibrato.
        f = F.interpolate(f0[:, None], size=samples, mode="linear", align_corners=False)[:, 0].clamp_min(0)
        v = F.interpolate(voiced[:, None], size=samples, mode="nearest")
        phase = torch.cumsum(2 * math.pi * f / self.sample_rate, dim=-1)
        source = sum(g * torch.sin(phase * (i + 1)) for i, g in enumerate(self.harmonic_gain))
        noise = torch.randn_like(source)
        if breath is None: breath = 1 - v[:, 0]
        else: breath = F.interpolate(breath[:, None], size=samples, mode="linear", align_corners=False)[:, 0]
        return (source * v[:, 0] + noise * breath.clamp(0, 1))[:, None]

class VocoderGenerator(nn.Module):
    def __init__(self, config: VocoderConfig = VocoderConfig()):
        super().__init__(); self.config = config
        cond_dim = config.n_mels + 5  # f0, vuv, energy, timing, breath
        self.condition = nn.Sequential(weight_norm(nn.Conv1d(cond_dim, config.channels, 7, padding=3)), nn.LeakyReLU(0.1), weight_norm(nn.Conv1d(config.channels, config.channels, 3, padding=1)))
        self.speaker = nn.Linear(config.speaker_dim, config.channels)
        self.style = nn.Linear(config.style_dim, config.channels)
        self.source = HarmonicSource(config.sample_rate)
        self.source_proj = weight_norm(nn.Conv1d(1, config.channels, 7, padding=3))
        self.ups = nn.ModuleList(); self.blocks = nn.ModuleList()
        ch = config.channels
        for rate, kernel in zip(config.upsample_rates, config.upsample_kernels):
            nxt = ch // 2
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, nxt, kernel, rate, padding=(kernel-rate)//2)))
            self.blocks.append(nn.ModuleList([ResBlock(nxt, k, d) for k, d in zip(config.resblock_kernels, config.resblock_dilations)]))
            ch = nxt
        self.output = nn.Sequential(nn.LeakyReLU(0.1), weight_norm(nn.Conv1d(ch, 1, 7, padding=3)), nn.Tanh())
    def forward(self, mel: Tensor, f0: Tensor, energy: Tensor, timing: Tensor | None = None, speaker: Tensor | None = None, style: Tensor | None = None, breath: Tensor | None = None) -> Tensor:
        b, _, frames = mel.shape
        if f0.shape != (b, frames) or energy.shape != (b, frames): raise ValueError("f0 and energy must be [batch, mel_frames]")
        timing = torch.zeros_like(f0) if timing is None else timing
        voiced = (f0 > 1).to(mel.dtype)
        breath = (1 - voiced) * 0.15 if breath is None else breath
        x = self.condition(torch.cat([mel, torch.log1p(f0)[:, None], voiced[:, None], energy[:, None], timing[:, None], breath[:, None]], 1))
        if speaker is not None: x = x + self.speaker(speaker).unsqueeze(-1)
        if style is not None: x = x + self.style(style).unsqueeze(-1)
        source = self.source(f0, voiced, breath, frames * self.config.hop_size)
        source = self.source_proj(source)
        for up, blocks in zip(self.ups, self.blocks):
            x = F.leaky_relu(up(F.leaky_relu(x, 0.1)), 0.1)
            x = sum(block(x) for block in blocks) / len(blocks)
        if x.shape[-1] != source.shape[-1]: x = F.interpolate(x, size=source.shape[-1], mode="linear", align_corners=False)
        return torch.tanh(self.output(x) + 0.05 * source[:, :1])

class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__(); self.period = period
        channels = [1, 32, 128, 512, 1024, 1024]
        self.layers = nn.ModuleList([weight_norm(nn.Conv2d(a,b,(5,1),(3,1),padding=(2,0))) for a,b in zip(channels,channels[1:])])
        self.out = weight_norm(nn.Conv2d(channels[-1], 1, (3,1), padding=(1,0)))
    def forward(self, x: Tensor):
        if x.shape[-1] % self.period: x = F.pad(x, (0, self.period - x.shape[-1] % self.period), mode="reflect")
        x = x.view(x.shape[0], 1, -1, self.period); feats=[]
        for layer in self.layers: x=F.leaky_relu(layer(x),.1); feats.append(x)
        return self.out(x).flatten(1), feats

class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__(); channels=[1,128,128,256,512,1024,1024]
        self.layers=nn.ModuleList([weight_norm(nn.Conv1d(a,b,15,1 if i==0 else 2,padding=7,groups=1 if i==0 else min(a,16))) for i,(a,b) in enumerate(zip(channels,channels[1:]))])
        self.out=weight_norm(nn.Conv1d(channels[-1],1,3,padding=1))
    def forward(self,x):
        feats=[]
        for layer in self.layers: x=F.leaky_relu(layer(x),.1); feats.append(x)
        return self.out(x).flatten(1),feats

class MultiDiscriminator(nn.Module):
    def __init__(self): super().__init__(); self.periods=nn.ModuleList([DiscriminatorP(p) for p in (2,3,5,7,11)]); self.scales=nn.ModuleList([DiscriminatorS() for _ in range(3)])
    def forward(self,x):
        results=[]
        for d in self.periods: results.append(d(x))
        for d in self.scales: results.append(d(x)); x=F.avg_pool1d(x,4,2,padding=2)
        return results

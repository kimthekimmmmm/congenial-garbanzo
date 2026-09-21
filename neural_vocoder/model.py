"""Source-filter GAN generator and periodicity-aware discriminators."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def wn(module: nn.Module) -> nn.Module:
    """Keep weight normalization centralized so it can be removed for deployment."""
    return weight_norm(module)


@dataclass
class VocoderConfig:
    sample_rate: int = 48_000
    hop_size: int = 300
    n_mels: int = 100
    channels: int = 256
    upsample_rates: tuple[int, ...] = (5, 5, 3, 4)
    upsample_kernels: tuple[int, ...] = (10, 10, 6, 8)
    resblock_kernels: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5),) * 3
    speaker_dim: int = 128
    style_dim: int = 128
    source_harmonics: int = 8

    def __post_init__(self) -> None:
        if self.sample_rate < 16_000 or self.hop_size <= 0:
            raise ValueError("sample_rate must be >= 16000 and hop_size must be positive")
        if math.prod(self.upsample_rates) != self.hop_size:
            raise ValueError("Product of upsample_rates must equal hop_size for sample-accurate alignment")
        if len(self.upsample_rates) != len(self.upsample_kernels) or self.channels % 2 ** len(self.upsample_rates):
            raise ValueError("upsample settings must match and channels must halve cleanly at every stage")


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: Sequence[int]):
        super().__init__()
        self.dilated = nn.ModuleList([wn(nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2)) for d in dilations])
        self.pointwise = nn.ModuleList([wn(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)) for _ in dilations])

    def forward(self, x: Tensor) -> Tensor:
        for dilated, pointwise in zip(self.dilated, self.pointwise):
            x = x + pointwise(F.leaky_relu(dilated(F.leaky_relu(x, 0.1)), 0.1))
        return x


class HarmonicSource(nn.Module):
    """Band-limited harmonic excitation with a separately controlled stochastic path.

    F0 is interpolated *before* integration, retaining glides and vibrato phase continuity.
    """
    def __init__(self, sample_rate: int, harmonics: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_amplitude = nn.Parameter(torch.full((harmonics,), 1.0 / harmonics))

    def forward(self, f0: Tensor, voiced: Tensor, breath: Tensor, samples: int) -> Tensor:
        f0_samples = F.interpolate(f0[:, None], size=samples, mode="linear", align_corners=False)[:, 0].clamp_min(0)
        voiced_samples = F.interpolate(voiced[:, None], size=samples, mode="nearest")[:, 0]
        breath_samples = F.interpolate(breath[:, None], size=samples, mode="linear", align_corners=False)[:, 0].clamp(0, 1)
        phase = torch.cumsum(2 * math.pi * f0_samples / self.sample_rate, dim=-1)
        # Mask harmonics past Nyquist to avoid deterministic aliasing on high notes.
        harmonic = torch.zeros_like(phase)
        for index, gain in enumerate(self.harmonic_amplitude, start=1):
            valid = (f0_samples * index < self.sample_rate / 2).to(phase.dtype)
            harmonic = harmonic + gain * torch.sin(index * phase) * valid
        noise = torch.randn_like(harmonic) * breath_samples
        return (harmonic * voiced_samples + noise)[:, None]


class VocoderGenerator(nn.Module):
    """Fast source-filter GAN generator with sample-aligned frame conditioning."""
    def __init__(self, config: VocoderConfig | None = None):
        super().__init__()
        self.config = config or VocoderConfig()
        c = self.config
        condition_channels = c.n_mels + 5  # log F0, V/UV, energy, boundary, breath
        self.condition = nn.Sequential(wn(nn.Conv1d(condition_channels, c.channels, 7, padding=3)), nn.LeakyReLU(0.1), wn(nn.Conv1d(c.channels, c.channels, 3, padding=1)))
        self.speaker = nn.Linear(c.speaker_dim, 2 * c.channels)
        self.style = nn.Linear(c.style_dim, 2 * c.channels)
        self.source = HarmonicSource(c.sample_rate, c.source_harmonics)
        self.ups, self.blocks, self.source_adapters = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        channels = c.channels
        for rate, kernel in zip(c.upsample_rates, c.upsample_kernels):
            next_channels = channels // 2
            self.ups.append(wn(nn.ConvTranspose1d(channels, next_channels, kernel, rate, padding=(kernel - rate) // 2)))
            self.blocks.append(nn.ModuleList([ResBlock(next_channels, k, ds) for k, ds in zip(c.resblock_kernels, c.resblock_dilations)]))
            # Inject the source at every resolution rather than only at the output.
            self.source_adapters.append(wn(nn.Conv1d(1, next_channels, 7, padding=3)))
            channels = next_channels
        self.output = wn(nn.Conv1d(channels, 1, 7, padding=3))

    def forward(self, mel: Tensor, f0: Tensor, energy: Tensor, timing: Tensor | None = None, speaker: Tensor | None = None, style: Tensor | None = None, breath: Tensor | None = None) -> Tensor:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, n_mels, frames]")
        batch, channels, frames = mel.shape
        if channels != self.config.n_mels or f0.shape != (batch, frames) or energy.shape != (batch, frames):
            raise ValueError("conditioning dimensions do not match configured n_mels and frame count")
        timing = torch.zeros_like(f0) if timing is None else timing
        if timing.shape != f0.shape:
            raise ValueError("timing must have shape [batch, frames]")
        voiced = (f0 > 1).to(mel.dtype)
        breath = (1 - voiced) * 0.15 if breath is None else breath.clamp(0, 1)
        if breath.shape != f0.shape:
            raise ValueError("breath must have shape [batch, frames]")
        x = self.condition(torch.cat((mel, torch.log1p(f0.clamp_min(0))[:, None], voiced[:, None], energy[:, None], timing[:, None], breath[:, None]), dim=1))
        for embedding, layer, name in ((speaker, self.speaker, "speaker"), (style, self.style, "style")):
            if embedding is not None:
                if embedding.shape != (batch, layer.in_features):
                    raise ValueError(f"{name} embedding has an invalid shape")
                scale, shift = layer(embedding).chunk(2, dim=1)
                x = x * (1 + scale[:, :, None]) + shift[:, :, None]
        source = self.source(f0, voiced, breath, frames * self.config.hop_size)
        for up, blocks, adapter in zip(self.ups, self.blocks, self.source_adapters):
            x = F.leaky_relu(up(F.leaky_relu(x, 0.1)), 0.1)
            at_scale = F.interpolate(source, size=x.shape[-1], mode="linear", align_corners=False)
            x = x + adapter(at_scale)
            x = sum(block(x) for block in blocks) / len(blocks)
        if x.shape[-1] != source.shape[-1]:
            x = F.interpolate(x, size=source.shape[-1], mode="linear", align_corners=False)
        return torch.tanh(self.output(F.leaky_relu(x, 0.1)) + 0.05 * source)

    def remove_weight_norm(self) -> None:
        for module in self.modules():
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__(); self.period = period
        pairs = zip((1, 32, 128, 512, 1024, 1024), (32, 128, 512, 1024, 1024, 1024))
        self.layers = nn.ModuleList([wn(nn.Conv2d(a, b, (5, 1), (3, 1), padding=(2, 0))) for a, b in pairs])
        self.output = wn(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: Tensor):
        pad = (-x.shape[-1]) % self.period
        if pad: x = F.pad(x, (0, pad), mode="reflect")
        x = x.view(x.shape[0], 1, -1, self.period); features = []
        for layer in self.layers: x = F.leaky_relu(layer(x), 0.1); features.append(x)
        return self.output(x).flatten(1), features


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        pairs = zip((1, 128, 128, 256, 512, 1024, 1024), (128, 128, 256, 512, 1024, 1024, 1024))
        self.layers = nn.ModuleList([wn(nn.Conv1d(a, b, 15, 1 if i == 0 else 2, padding=7, groups=1 if i == 0 else min(a, 16))) for i, (a, b) in enumerate(pairs)])
        self.output = wn(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x: Tensor):
        features = []
        for layer in self.layers: x = F.leaky_relu(layer(x), 0.1); features.append(x)
        return self.output(x).flatten(1), features


class MultiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = nn.ModuleList([DiscriminatorP(p) for p in (2, 3, 5, 7, 11)])
        self.scales = nn.ModuleList([DiscriminatorS() for _ in range(3)])

    def forward(self, x: Tensor):
        results = [d(x) for d in self.periods]
        for discriminator in self.scales:
            results.append(discriminator(x))
            x = F.avg_pool1d(x, 4, 2, padding=2)
        return results

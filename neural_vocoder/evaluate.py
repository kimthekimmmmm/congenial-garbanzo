"""Objective regression checks for common vocoder failures."""
from __future__ import annotations
import torch
from .data import extract_features


def metrics(reference: torch.Tensor, generated: torch.Tensor, sample_rate: int = 48_000, hop_size: int = 300, n_mels: int = 100) -> dict[str, float | bool]:
    ref, gen = reference.squeeze(), generated.squeeze()
    if ref.ndim != 1 or gen.ndim != 1:
        raise ValueError("reference and generated audio must be mono waveforms")
    length = min(len(ref), len(gen))
    if length < 2 * hop_size:
        raise ValueError("audio is too short for objective evaluation")
    ref, gen = ref[:length], gen[:length]
    a, b = extract_features(ref, sample_rate, hop_size, n_mels), extract_features(gen, sample_rate, hop_size, n_mels)
    voiced = (a["f0"] > 1) & (b["f0"] > 1)
    cents = (1200 * torch.log2((b["f0"][voiced] + 1e-6) / (a["f0"][voiced] + 1e-6))).abs().mean() if voiced.any() else torch.tensor(float("nan"))
    spectral_distance = (a["mel"] - b["mel"]).square().mean().sqrt()
    transient_error = (torch.diff(a["energy"]) - torch.diff(b["energy"])).abs().mean()
    spectrum = torch.fft.rfft(gen).abs().clamp_min(1e-8)
    high_band_ratio = spectrum[int(0.75 * len(spectrum)):].sum() / spectrum.sum()
    flatness = torch.exp(torch.log(spectrum).mean()) / spectrum.mean()
    clipping_ratio = (gen.abs() > 0.995).float().mean()
    return {
        "f0_cents": float(cents), "log_spectral_distance": float(spectral_distance),
        "transient_error": float(transient_error), "peak": float(gen.abs().max()),
        "high_band_ratio": float(high_band_ratio), "spectral_flatness": float(flatness),
        "clipping_ratio": float(clipping_ratio),
        "artifact_warning": bool(clipping_ratio > 0.001 or high_band_ratio > 0.35 or flatness > 0.65),
    }

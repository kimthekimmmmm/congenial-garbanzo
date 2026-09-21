from __future__ import annotations
import torch
from .data import extract_features

def metrics(reference, generated, sample_rate=48000, hop_size=300, n_mels=100):
    ref=reference.squeeze(); gen=generated.squeeze(); n=min(len(ref),len(gen)); ref,gen=ref[:n],gen[:n]
    a=extract_features(ref,sample_rate,hop_size,n_mels); b=extract_features(gen,sample_rate,hop_size,n_mels)
    voiced=(a['f0']>1)&(b['f0']>1); cents=(1200*torch.log2((b['f0'][voiced]+1e-6)/(a['f0'][voiced]+1e-6))).abs().mean() if voiced.any() else torch.tensor(float('nan'))
    lsd=(a['mel']-b['mel']).square().mean().sqrt(); onset=(torch.diff(a['energy'])-torch.diff(b['energy'])).abs().mean()
    high=torch.fft.rfft(gen).abs(); high_ratio=high[int(.75*len(high)):].sum()/(high.sum()+1e-8)
    return {"f0_cents":float(cents),"log_spectral_distance":float(lsd),"transient_error":float(onset),"peak":float(gen.abs().max()),"high_band_ratio":float(high_ratio),"artifact_warning":bool(gen.abs().max()>.999 or high_ratio>.35)}

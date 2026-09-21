from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
LOG = logging.getLogger(__name__)

def load_audio(path: str, sample_rate: int) -> Tensor:
    wav, sr = sf.read(path, always_2d=True, dtype="float32"); wav = torch.from_numpy(wav.mean(1))
    if sr != sample_rate:
        wav = F.interpolate(wav[None,None], size=round(len(wav)*sample_rate/sr), mode="linear", align_corners=False)[0,0]
    return wav.clamp(-1, 1)

def mel_filter(sample_rate: int, n_fft: int, n_mels: int, device: torch.device) -> Tensor:
    hz = torch.linspace(0, sample_rate/2, n_fft//2+1, device=device); m=lambda x: 2595*torch.log10(1+x/700); hz2=lambda x:700*(10**(x/2595)-1)
    points=hz2(torch.linspace(m(torch.tensor(0.)),m(torch.tensor(sample_rate/2.)),n_mels+2,device=device)); bins=torch.searchsorted(hz,points).clamp(0,len(hz)-1)
    fb=torch.zeros(n_mels,len(hz),device=device)
    for i in range(n_mels):
        a,b,c=bins[i:i+3]
        if b>a: fb[i,a:b]=(hz[a:b]-hz[a])/(hz[b]-hz[a])
        if c>b: fb[i,b:c]=(hz[c]-hz[b])/(hz[c]-hz[b])
    return fb

def extract_features(wav: Tensor, sample_rate: int, hop_size: int, n_mels: int) -> dict[str, Tensor]:
    n_fft=2048; window=torch.hann_window(n_fft,device=wav.device)
    spec=torch.stft(wav,n_fft,hop_size,n_fft,window=window,return_complex=True,center=True).abs()
    mel=torch.log(torch.clamp(mel_filter(sample_rate,n_fft,n_mels,wav.device) @ spec,min=1e-5))
    energy=torch.sqrt(F.avg_pool1d(wav.square()[None,None], hop_size, hop_size, ceil_mode=True)[0,0]+1e-8)[:mel.shape[-1]]
    # Lightweight autocorrelation baseline. Replace with RMVPE/CREPE for training-quality labels.
    f0=torch.zeros(mel.shape[-1],device=wav.device); minlag=max(1,sample_rate//800); maxlag=sample_rate//50
    padded=F.pad(wav,(n_fft//2,n_fft//2));
    for i in range(len(f0)):
        frame=padded[i*hop_size:i*hop_size+n_fft]; frame=frame-frame.mean(); ac=F.conv1d(frame[None,None],frame.flip(0)[None,None],padding=n_fft-1)[0,0,n_fft-1:]
        hi=min(maxlag,len(ac)-1); peak=minlag+torch.argmax(ac[minlag:hi]).item()
        if ac[peak] > .3*ac[0]: f0[i]=sample_rate/peak
    return {"mel":mel, "f0":f0, "energy":energy, "timing":torch.zeros_like(f0)}

def prepare(manifest: str, output: str, sample_rate: int, hop_size: int, n_mels: int) -> None:
    out=Path(output); out.mkdir(parents=True,exist_ok=True); records=[]
    for line in Path(manifest).read_text().splitlines():
        item=json.loads(line); wav=load_audio(item["audio"],sample_rate); feats=extract_features(wav,sample_rate,hop_size,n_mels)
        # Boundary impulses make phoneme attacks explicit while preserving frame alignment.
        for boundary in item.get("phoneme_boundaries", []):
            frame = round(float(boundary) * sample_rate / hop_size)
            if 0 <= frame < feats["timing"].numel(): feats["timing"][frame] = 1.0
        name=f"{len(records):07d}.npz"; np.savez_compressed(out/name,audio=wav.numpy(),**{k:v.numpy() for k,v in feats.items()})
        records.append({"features":str(out/name),"speaker":item.get("speaker",0),"style":item.get("style",0)})
    (out/"manifest.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records)); LOG.info("Prepared %d examples in %s",len(records),out)

class FeatureDataset(Dataset):
    def __init__(self, manifest: str, segment_frames: int, hop_size: int, speaker_dim: int=128, style_dim: int=128):
        self.rows=[json.loads(x) for x in Path(manifest).read_text().splitlines()]; self.frames=segment_frames; self.hop=hop_size; self.speaker_dim=speaker_dim; self.style_dim=style_dim
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; x=np.load(row["features"]); total=x["mel"].shape[-1]; start=np.random.randint(0,max(1,total-self.frames+1)); end=min(total,start+self.frames)
        def pad(a,n): return np.pad(a,[(0,0)]*(a.ndim-1)+[(0,max(0,n-a.shape[-1]))])
        result={k:torch.from_numpy(pad(x[k][...,start:end],self.frames)).float() for k in ("mel","f0","energy","timing")}
        audio=x["audio"][start*self.hop:(start+self.frames)*self.hop]; result["audio"]=torch.from_numpy(np.pad(audio,(0,max(0,self.frames*self.hop-len(audio))))).float()[None]
        result["speaker"]=F.one_hot(torch.tensor(int(row["speaker"])%self.speaker_dim),self.speaker_dim).float(); result["style"]=F.one_hot(torch.tensor(int(row["style"])%self.style_dim),self.style_dim).float()
        return result

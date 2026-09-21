from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F

def stft_loss(real: Tensor, fake: Tensor) -> Tensor:
    total=0.
    for n in (512,1024,2048):
        w=torch.hann_window(n,device=real.device); a=torch.stft(real[:,0],n,n//4,n,window=w,return_complex=True).abs(); b=torch.stft(fake[:,0],n,n//4,n,window=w,return_complex=True).abs()
        total += (a-b).abs().mean()/(a.mean().detach()+1e-6) + (torch.log(a+1e-7)-torch.log(b+1e-7)).abs().mean()
    return total/3

def discriminator_loss(real_out, fake_out): return sum((1-r).square().mean()+f.square().mean() for (r,_),(f,_) in zip(real_out,fake_out))
def generator_loss(fake_out): return sum((1-f).square().mean() for f,_ in fake_out)
def feature_matching(real_out,fake_out): return sum(sum((a.detach()-b).abs().mean() for a,b in zip(rf,ff)) for (_,rf),(_,ff) in zip(real_out,fake_out))
def generator_objective(real: Tensor, fake: Tensor, real_d, fake_d):
    wave=F.l1_loss(fake,real); spectral=stft_loss(real,fake); adv=generator_loss(fake_d); fm=feature_matching(real_d,fake_d)
    return wave + 45*spectral + 2*adv + 2*fm, {"wave":wave,"stft":spectral,"adv":adv,"fm":fm}

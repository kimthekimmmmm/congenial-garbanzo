import torch
from neural_vocoder.model import VocoderConfig,VocoderGenerator
from neural_vocoder.losses import stft_loss
def test_generator_is_sample_aligned_and_bounded():
 c=VocoderConfig(sample_rate=24000,hop_size=24,n_mels=8,channels=32,upsample_rates=(2,2,2,3),upsample_kernels=(4,4,4,6),resblock_kernels=(3,),resblock_dilations=((1,3),),speaker_dim=4,style_dim=4)
 m=VocoderGenerator(c).eval(); frames=5
 with torch.no_grad(): y=m(torch.randn(1,8,frames),torch.tensor([[200.,201.,0.,220.,220.]]),torch.ones(1,frames),speaker=torch.zeros(1,4),style=torch.zeros(1,4))
 assert y.shape==(1,1,frames*24); assert y.abs().max()<=1.1
def test_spectral_loss_identity_is_small():
 x=torch.randn(1,1,4096)*.1
 assert stft_loss(x,x)<1e-6

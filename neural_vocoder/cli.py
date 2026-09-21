from __future__ import annotations
import argparse,json,logging
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from .data import prepare,load_audio
from .model import VocoderConfig,VocoderGenerator
from .train import train
from .evaluate import metrics
def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False); model=VocoderGenerator(VocoderConfig(**ck['config'])).to(device).eval(); model.load_state_dict(ck['model']); return model
def main():
 p=argparse.ArgumentParser(); p.add_argument('--verbose',action='store_true'); sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare'); a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--sample-rate',type=int,default=48000);a.add_argument('--hop-size',type=int,default=300);a.add_argument('--n-mels',type=int,default=100)
 a=sub.add_parser('train'); a.add_argument('--config',required=True);a.add_argument('--train-manifest',required=True);a.add_argument('--output',default='runs/latest')
 a=sub.add_parser('infer'); a.add_argument('--checkpoint',required=True);a.add_argument('--features',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default=None)
 a=sub.add_parser('export');a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True)
 a=sub.add_parser('evaluate');a.add_argument('--reference',required=True);a.add_argument('--generated',required=True);a.add_argument('--sample-rate',type=int,default=48000);a.add_argument('--hop-size',type=int,default=300)
 x=p.parse_args(); logging.basicConfig(level=logging.DEBUG if x.verbose else logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 if x.command=='prepare': prepare(x.manifest,x.output,x.sample_rate,x.hop_size,x.n_mels)
 elif x.command=='train': train(yaml.safe_load(Path(x.config).read_text()),x.train_manifest,x.output)
 elif x.command=='infer':
  dev=torch.device(x.device or ('cuda' if torch.cuda.is_available() else 'cpu')); m=load_model(x.checkpoint,dev); z=np.load(x.features); mel=torch.from_numpy(z['mel']).float()[None].to(dev); f0=torch.from_numpy(z['f0']).float()[None].to(dev); e=torch.from_numpy(z['energy']).float()[None].to(dev); t=torch.from_numpy(z.get('timing',np.zeros_like(z['f0']))).float()[None].to(dev)
  with torch.inference_mode(): y=m(mel,f0,e,t).squeeze().cpu().numpy()
  sf.write(x.output,y,m.config.sample_rate)
 elif x.command=='export':
  m=load_model(x.checkpoint,torch.device('cpu')); c=m.config; ex=(torch.zeros(1,c.n_mels,8),torch.ones(1,8)*220,torch.ones(1,8)*.1)
  torch.jit.trace(m,ex,check_trace=False).save(x.output)
 else:
  r=load_audio(x.reference,x.sample_rate);g=load_audio(x.generated,x.sample_rate); print(json.dumps(metrics(r,g,x.sample_rate,x.hop_size),indent=2))
if __name__=='__main__': main()

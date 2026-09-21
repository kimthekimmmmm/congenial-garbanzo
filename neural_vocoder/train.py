from __future__ import annotations
import logging, os, torch
from torch.utils.data import DataLoader
from .model import VocoderConfig,VocoderGenerator,MultiDiscriminator
from .data import FeatureDataset
from .losses import discriminator_loss,generator_objective
LOG=logging.getLogger(__name__)
def train(cfg: dict, manifest: str, output: str):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); mc=VocoderConfig(**{k:cfg[k] for k in VocoderConfig.__dataclass_fields__ if k in cfg})
    g=VocoderGenerator(mc).to(dev); d=MultiDiscriminator().to(dev); ds=FeatureDataset(manifest,cfg['segment_frames'],mc.hop_size,mc.speaker_dim,mc.style_dim); dl=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg.get('num_workers',0),pin_memory=dev.type=='cuda',drop_last=True)
    og=torch.optim.AdamW(g.parameters(),cfg['learning_rate'],betas=(.8,.99)); od=torch.optim.AdamW(d.parameters(),cfg['learning_rate'],betas=(.8,.99)); os.makedirs(output,exist_ok=True); step=0
    while step<cfg['max_steps']:
      for batch in dl:
        batch={k:v.to(dev) for k,v in batch.items()}; fake=g(batch['mel'],batch['f0'],batch['energy'],batch['timing'],batch['speaker'],batch['style']); real=batch['audio']
        if step>=cfg.get('discriminator_warmup_steps',0):
          od.zero_grad(); ld=discriminator_loss(d(real),d(fake.detach())); ld.backward(); od.step()
        og.zero_grad(); ro=d(real); fo=d(fake); lg,parts=generator_objective(real,fake,ro,fo); lg.backward(); torch.nn.utils.clip_grad_norm_(g.parameters(),10); og.step()
        if step%cfg.get('log_interval',20)==0: LOG.info('step=%d generator=%.3f %s',step,lg.item(),{k:round(v.item(),3) for k,v in parts.items()})
        if step and step%cfg.get('checkpoint_interval',5000)==0: torch.save({'config':mc.__dict__,'model':g.state_dict(),'step':step},f'{output}/generator-{step}.pt')
        step+=1
        if step>=cfg['max_steps']: break

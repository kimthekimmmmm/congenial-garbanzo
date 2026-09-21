"""Training loop with AMP, reproducible checkpoints, and validation reporting."""
from __future__ import annotations
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import FeatureDataset
from .losses import discriminator_loss, generator_objective, stft_loss
from .model import MultiDiscriminator, VocoderConfig, VocoderGenerator

LOG = logging.getLogger(__name__)


def _loader(manifest: str, cfg: dict, model_cfg: VocoderConfig, device: torch.device, shuffle: bool) -> DataLoader:
    dataset = FeatureDataset(manifest, cfg["segment_frames"], model_cfg.hop_size, model_cfg.speaker_dim, model_cfg.style_dim)
    if not len(dataset):
        raise ValueError(f"dataset manifest is empty: {manifest}")
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg.get("num_workers", 0), pin_memory=device.type == "cuda", drop_last=shuffle, persistent_workers=cfg.get("num_workers", 0) > 0)


def _synthesize(generator: VocoderGenerator, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return generator(batch["mel"], batch["f0"], batch["energy"], batch["timing"], batch["speaker"], batch["style"])


def _checkpoint(path: Path, generator: VocoderGenerator, discriminator: MultiDiscriminator, optimizer_g, optimizer_d, model_cfg: VocoderConfig, step: int) -> None:
    torch.save({"config": model_cfg.__dict__, "model": generator.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "step": step}, path)


@torch.no_grad()
def validate(generator: VocoderGenerator, loader: DataLoader, device: torch.device, batches: int = 8) -> float:
    generator.eval(); values = []
    for index, batch in enumerate(loader):
        if index >= batches: break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        values.append(stft_loss(batch["audio"], _synthesize(generator, batch)).item())
    generator.train()
    return sum(values) / max(len(values), 1)


def train(cfg: dict, manifest: str, output: str, valid_manifest: str | None = None, resume: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = VocoderConfig(**{key: cfg[key] for key in VocoderConfig.__dataclass_fields__ if key in cfg})
    generator, discriminator = VocoderGenerator(model_cfg).to(device), MultiDiscriminator().to(device)
    train_loader = _loader(manifest, cfg, model_cfg, device, shuffle=True)
    valid_loader = _loader(valid_manifest, cfg, model_cfg, device, shuffle=False) if valid_manifest else None
    optimizer_g = torch.optim.AdamW(generator.parameters(), cfg["learning_rate"], betas=(0.8, 0.99))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), cfg["learning_rate"], betas=(0.8, 0.99))
    start = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        generator.load_state_dict(state["model"]); discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"]); optimizer_d.load_state_dict(state["optimizer_d"]); start = int(state["step"])
        LOG.info("resumed checkpoint %s at step %d", resume, start)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    amp_enabled = device.type == "cuda" and cfg.get("mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    step = start
    LOG.info("training on %s; AMP=%s; %d examples", device, amp_enabled, len(train_loader.dataset))
    while step < cfg["max_steps"]:
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}; real = batch["audio"]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                fake = _synthesize(generator, batch)
                if step >= cfg.get("discriminator_warmup_steps", 0):
                    d_loss = discriminator_loss(discriminator(real), discriminator(fake.detach()))
                else: d_loss = torch.zeros((), device=device)
            if step >= cfg.get("discriminator_warmup_steps", 0):
                optimizer_d.zero_grad(set_to_none=True); scaler.scale(d_loss).backward(); scaler.step(optimizer_d)
            for parameter in discriminator.parameters(): parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                real_scores, fake_scores = discriminator(real), discriminator(fake)
                g_loss, pieces = generator_objective(real, fake, real_scores, fake_scores)
            scaler.scale(g_loss).backward(); scaler.unscale_(optimizer_g); torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0); scaler.step(optimizer_g); scaler.update()
            for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            if step % cfg.get("log_interval", 20) == 0:
                LOG.info("step=%d g=%.3f d=%.3f %s", step, g_loss.item(), d_loss.item(), {k: round(v.item(), 3) for k, v in pieces.items()})
            if valid_loader and step and step % cfg.get("valid_interval", 2_000) == 0:
                LOG.info("step=%d validation_stft=%.4f", step, validate(generator, valid_loader, device))
            if step and step % cfg.get("checkpoint_interval", 5_000) == 0:
                _checkpoint(out / f"checkpoint-{step}.pt", generator, discriminator, optimizer_g, optimizer_d, model_cfg, step)
            step += 1
            if step >= cfg["max_steps"]: break
    _checkpoint(out / "last.pt", generator, discriminator, optimizer_g, optimizer_d, model_cfg, step)

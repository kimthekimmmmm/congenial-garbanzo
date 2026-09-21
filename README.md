# Aural Vocoder

A compact, production-oriented neural waveform vocoder for expressive speech and singing. It uses a HiFi-GAN-style multi-receptive-field generator, explicit pitch/energy/noise controls, and multi-period/multi-scale discriminators. It is deliberately small enough to run locally, while exposing the conditioning and losses required for a serious training run.

## Design and risks

**Architecture.** A frame-rate conditioning encoder fuses log-mel, continuous F0 (in Hz), voiced/unvoiced flag, energy, phoneme-boundary timing, speaker identity, style, and optional breath/noise. A voiced harmonic excitation and explicitly controlled stochastic breath/noise are injected at every upsampling scale. Multi-receptive-field residual blocks protect both long, stable singing harmonics and short consonant/transient detail. The default 48 kHz / 300-hop configuration has an exact 300x upsampling ratio.

**Why this design.** Diffusion waveform generators can be excellent but are often too slow for practical local singing iteration. Adversarial GAN vocoders are fast and stable once trained; explicit source conditioning materially reduces F0 drift on long notes. The discriminators operate at several periods and scales, so pitch periodicity, transients, and broadband fricatives receive separate pressure.

**Main risks / mitigations.** (1) F0 extraction errors cause buzz: use continuous F0 plus V/UV labels and audited F0 extraction. (2) adversarial collapse causes metallic fizz: use feature matching, multi-resolution STFT, waveform loss, and discriminator warm-up. (3) boundary blur: pass a timing channel and crop aligned waveform/conditioning segments. (4) singing exposes periodic artifacts: include sustained notes and vibrato in training, and measure cents error. High realism depends substantially on clean, licensed paired training data; this repository supplies the implementation, not pretrained voice weights.

## Quick start

```bash
pip install -e '.[train,test]'
aural-vocoder prepare --manifest data/manifest.jsonl --output data/prepared --sample-rate 48000 --hop-size 300
# inspect and edit configs/default.yaml, then:
aural-vocoder train --config configs/default.yaml --train-manifest data/prepared/train.jsonl --valid-manifest data/prepared/valid.jsonl --output runs/latest
aural-vocoder infer --checkpoint runs/latest/last.pt --features example.npz --output output.wav
aural-vocoder export --checkpoint runs/latest/last.pt --output generator.ts
```

The preparation manifest is JSONL with `audio`, optional `speaker`, `style`, and optional `phoneme_boundaries` (seconds). It writes `.npz` features containing `mel`, `f0`, `energy`, and `timing`. For best results, extract reliable F0 externally (RMVPE/CREPE) and replace `f0` in the prepared files; the built-in autocorrelation extractor is a dependency-free baseline.

## Training operations

Training supports CUDA automatic mixed precision, resumable checkpoints, a discriminator warm-up, gradient clipping, and optional held-out validation: `--valid-manifest data/prepared/valid.jsonl`. The final checkpoint is written to `runs/latest/last.pt`; use `--resume` with a checkpoint to continue an interrupted run.

## Quality validation

`aural-vocoder evaluate` reports F0 cents error over voiced frames, log-spectral distance, onset/transient error, clipping, high-band energy ratio, and spectral-flatness warnings. These are regression guards, not a substitute for listening tests. Train with diverse speech/singing, multiple dynamics, sibilants, breaths, and sustained notes; reserve singers, lyrics, and pitches for validation.

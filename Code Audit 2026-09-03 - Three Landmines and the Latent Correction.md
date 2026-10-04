---
tags: [lewm, talking-head, jepa, audit, correction, methodology]
created: 2026-09-03
status: active
supersedes-claims-in:
  - "[[FaRL Verification and Decoder Plan]]"
  - "[[JEPA Talking Head - Verified Reference Data]]"
  - "[[Next Steps - Single-Identity JEPA Talking Head]]"
---

# Code Audit 2026-09-03 — Three Landmines and the Latent Correction

Found by reading `talking_head_jepa.py`, `JEPA_TalkingHead_EncoderRetrain.ipynb` and
`JEPA_Latent_Probing.ipynb` line by line while building the Colab notebooks. All three
are verified against source, not inferred. Two of them invalidate claims already filed
in this vault.

Related: [[JEPA Talking Head - Study Plan]] · [[LeWM Talking-Head JEPA]]

---

## 1. The FaRL latent is 768-d, not 196×768 — a 196× compression, not 1.0

**This corrects a load-bearing claim in three notes.** They state:

> 196 tokens × 768 dims = 150,528 floats = exactly a 224² image's 150,528 values.
> Ratio exactly 1.0. No compression bottleneck.

That arithmetic is right about ViT-B/16's *token grid*. It is wrong about this
project's pipeline. `FarlFrozenFaceEncoder.forward()`
(`talking_head_jepa.py`, 2026-08-26) ends:

```python
x = self.visual.ln_post(x)                 # applied to ALL tokens, not just x[:, 0]
patch_mean = x[:, 1:].mean(dim=1).float()  # (N, 768)
...
out.last_hidden_state = patch_mean.unsqueeze(1)   # (N, 1, 768)
```

The 196 tokens are **mean-pooled** into one 768-d vector before anything downstream
sees them. `JEPA_Latent_Probing.ipynb` cells 65/73/89 do the same thing
(`hidden[:, 1:].mean(dim=1)`, printed as "patch-mean"), so `farl_patch_raw` in the
probing results is also the pooled vector, not the dense map.

The encoder's own docstring says "dense patch-token mean" — accurate. My note titles
turned that into "dense patch tokens", and the error propagated.

**Actual compression ratios:**

| representation | dims/frame | vs 224²×3 = 150,528 | who uses it |
|---|---|---|---|
| `pooled_state` (`projector` output) | 256 | **588×** | the JEPA predictor, Run D/E |
| `farl_patch_mean` (encoder output) | 768 | **196×** | `FarlFrozenFaceEncoder` |
| `farl_dense` (the assumed one) | 150,528 | 1.0× | nothing, yet |

### What this changes

- **The decoder is in the pooled-vector regime, not the spatial-token regime.** RAE's
  rFID numbers (DINOv2-B 0.49, SigLIP2-B 0.53, MAE-B 0.16 vs SD-VAE 0.62) are for
  *patch tokens*. They do not transfer to a 768-d pooled vector, so the "your proof of
  invertibility is very likely to succeed" conclusion in
  [[FaRL Verification and Decoder Plan]] §5.1 is not supported by that citation for the
  pipeline as built. The relevant precedents for a pooled vector are RCDM and unCLIP.
- **Path 1 (MAE-style decoder on patch tokens) does not apply as written.** It needs
  dense tokens the pipeline does not produce.
- **Path 0 (per-patch ridge, token → its own 16×16×3 block) is still exactly right** —
  but only for `farl_dense`, so extracting dense tokens is now a prerequisite for it
  rather than a free day-one check.

### The one thing that survives intact

The 512-d identity-template inversion result ([arXiv:1703.00832](https://arxiv.org/abs/1703.00832):
95.20% TAR @ 0.1% FAR on LFW from a template that discards pose, expression and
lighting by design) is a *pooled-vector* result. It is the right precedent, and it
argues 768-d is plausibly invertible. The 196×768 framing was never needed for it.

### The fix, if pooling is the bottleneck

One line: `return x[:, 1:]` instead of its mean, plus a projector that accepts
196×768. `S1_Decoder_Oracle.ipynb` measures all three representations under identical
budgets so this is decided by measurement rather than assumption. A result where dense
tokens invert well and the patch-mean does not is the *good* outcome: a concrete,
one-line, well-motivated fix.

---

## 2. Run D and Run E had no train/test split at all

`SyncTalkDatasetFixed` (`JEPA_TalkingHead_EncoderRetrain.ipynb`):

```python
def __len__(self):
    return len(self.image_paths) - self.seq_len
```

with `DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)`.
Every window in the clip is a training sample. `May`'s official SyncTalk split —
`transforms_train.json` frames 0–5519, `transforms_val.json` frames **5520–6072** —
was never applied.

**Consequence: every rollout number reported for Run D/E on `May` is in-sample.** That
includes the Part 8 compounding-error curve, the Part 9d epoch selection, and the
0.051 / 0.112 plateau figures. They are not wrong as *training* diagnostics; they are
simply not evidence about generalisation, and the "promising finding" framing rested
on treating them as such.

This also means the earlier concern about *random-split leakage* (test→nearest-train
mean 2.52 frames, max 7, 100% within 280 ms) understated the problem. There was no
split to leak across.

**Fix:** `S0_Instrument_And_Audio_Diagnostics.ipynb` retrains the predictor side
("Run F") on frames 0–4966, selects the epoch on 4967–5519, and touches 5520–6072 only
in the verdict cell. It reports the same metric in-sample *and* held-out, side by side,
because the gap between them is the diagnostic.

---

## 3. `TalkingHeadJEPA.forward()` mis-aligns audio by four frames

`talking_head_jepa.py`:

```python
ctx_emb = emb[:, :ctx_len]
tgt_emb = emb[:, num_preds:]          # frames 4..T
pred_emb = self.jepa.predict(ctx_emb, ctx_act)   # ctx_act = act_emb[:, :ctx_len]
```

The predictor is asked for frames 4–7 while conditioned on the audio of frames 0–3.
Since each `aud_ave` row covers a 0.2 s window and 4 frames is 160 ms, the condition
barely overlaps the frames being predicted. A model trained this way could not learn
lip-sync even in principle.

**Run D/E are not affected.** The `EncoderRetrain` notebook does not call
`model.forward()`; it uses its own loss with the correct alignment:

```python
pred_act = act_emb[:, ctx_len : ctx_len + num_preds]   # this hop's FUTURE audio
tgt_emb  = emb[:, ctx_len : ctx_len + num_preds].detach()
```

So this is a landmine rather than an active bug — but `forward()` is the obvious
method for anyone (including a future you, or a collaborator) to call. **Delete it or
make it raise.** Two implementations of the same loss with different alignment, one of
them silently wrong, is exactly how a result gets published and then retracted.

---

## 4. Two bugs found by smoke-testing the new notebooks

Both were caught by executing the notebooks' own cells against synthetic data on CPU
(`diagnostics/smoke_nb1.py`, `smoke_nb2.py`), which is worth keeping as a habit.

**The `audio_embedder` gets exactly zero gradient on optimizer step 0.** AdaLN-zero
zero-initialises `adaLN_modulation[-1]`'s weight *and* bias, so the gradient with
respect to the condition is `Wᵀ·grad = 0`. The audio path is literally disconnected at
init and reconnects on step 1, once the modulation weight has moved off zero (its own
gradient, `grad_out · condᵀ`, is non-zero). So "audio does nothing" at epoch 0 is
expected and is not evidence of anything.

**The N-way matching harness was biased toward the correct answer.** With the true
candidate always at position 0 of the candidate list, `np.argmin`'s first-wins
tie-breaking hands every tie to the correct answer — which drove a *mean-image control*
to 3× chance on synthetic data. The true candidate is now placed at a random position,
and the notebook runs a self-test (perfect reconstruction must be near ceiling, mean
image must sit at chance) before reporting any decoder number.

This one matters retrospectively: **if the earlier diffusion-decoder DDIM check used
first-position ordering, its 1/300 was measured on a harness biased in its favour**,
which makes that result worse, not better. Worth checking `JEPA_TalkingHead_GenLatent.ipynb`
before citing the 1/300 anywhere.

---

## 5. Corrected split reference for `May`

Read from the identity's own transform files, not derived:

| | frames | count |
|---|---|---|
| `transforms_train.json` | 0–5519 | 5520 |
| `transforms_val.json` | 5520–6072 | 553 |
| `gt_imgs/` on disk | 0–6072 | 6073 |
| `aud_ave.npy` | — | 6072 × 512 |

`int(6073 × 10/11) = 5520`, so the `10/11` convention and the transform files agree
here. The earlier note of "train 0–5529 / test 5530–6082" was off by ten frames at the
boundary; use 5520 / 5519.

---

## 6. The one methodological change worth keeping

The face encoder is frozen, so its output is a pure function of the frame and can be
cached once for all 6073 frames. Everything that trains (`projector`,
`audio_embedder`, `predictor`, `pred_proj`) then trains on 768-d vectors.

**A Run-E-equivalent 15-epoch run goes from ~4 GPU-hours to about two minutes.**

That is not just convenience. Pre-registered controls, a shuffled-audio null, an
in-sample/held-out pair, and a three-step Tier-0 fix ablation are all affordable only
if a run costs minutes. The reason this project has been reasoning from single
uncontrolled runs is that each one cost half a day.

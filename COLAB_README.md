# Colab notebooks — single-identity JEPA talking head (`May`)

Two notebooks, run in order. Both are self-contained apart from the repo modules
(`module.py`, `jepa.py`) and `data/May_train.tar`, which must be on Drive at
`/content/drive/MyDrive/Data/le-wm` (change `CODE_BASE` in cell 2 if not).

| notebook | question | runtime |
|---|---|---|
| `S0_Instrument_And_Audio_Diagnostics.ipynb` | does the predictor use audio, on frames it has never seen? | ~15 min on a T4 (FaRL cache ~10 min once, then seconds/epoch) |
| `S1_Decoder_Oracle.ipynb` | can the latent be inverted to RGB at all? | ~45 min on a T4 |

Both write a **pre-registration** JSON to `prereg/` before computing anything, and a
**result** JSON to `study_results/` at the end. The pair is the artifact; either alone
is worth much less. The pre-registration cell is write-once — re-running cannot move a
goalpost, and changing one requires deleting the file by hand.

## Three problems in the existing code that these fix

1. **Run D/E had no train/test split.** `SyncTalkDatasetFixed.__len__` is
   `len(image_paths) - seq_len` and the loader shuffles over all of it. `May`'s
   official validation frames (5520–6072, per `transforms_val.json`) were in the
   training set, so every reported rollout number for Run D/E on `May` is in-sample.
   S0 retrains on frames 0–4966 with 4967–5519 for epoch selection, and touches
   5520–6072 only in the verdict cell.

2. **`TalkingHeadJEPA.forward()` mis-aligns audio.** It uses
   `ctx_act = act_emb[:, :ctx_len]` with `tgt_emb = emb[:, num_preds:]` — predicting
   frames 4–7 while conditioned on the audio of frames 0–3. `JEPA_TalkingHead_EncoderRetrain.ipynb`
   does *not* call it (it uses `pred_act = act_emb[:, ctx_len:ctx_len+num_preds]`, the
   correct future audio), so Run D/E are unaffected — but the method is a landmine for
   anyone who calls it. Consider deleting it or asserting on it.

3. **The latent is 768-d, not 196×768.** `FarlFrozenFaceEncoder.forward()` returns
   `x[:, 1:].mean(dim=1)`. The "196 tokens × 768 = 150,528 floats = no compression
   bottleneck" claim in four filed notes describes a pipeline that does not exist —
   the real ratio is **196× compression** for the encoder output and **588×** for the
   256-d `pooled_state` the predictor actually uses. S1 measures both, plus dense
   tokens, so the comparison settles it instead of assuming.

## The change that makes iteration cheap

The face encoder is frozen, so its output is a pure function of the frame. Cache it
once for all 6073 frames and everything that trains (`projector`, `audio_embedder`,
`predictor`, `pred_proj`) trains on 768-d vectors. A Run-E-equivalent 15-epoch run
goes from ~4 GPU-hours to a couple of minutes. That is what makes running the
pre-registered controls — and the Tier-0 fix ablation — affordable at all.

## Offline smoke tests

`smoke_nb1.py` / `smoke_nb2.py` execute the notebooks' own cells against synthetic
data on CPU, so a bug they catch is a bug in the notebook rather than in a copy.
They need `torch`, `einops`, `scipy`, `statsmodels`, `opencv-python` and the repo on
`sys.path`. Run them after editing either notebook.

They already caught two real problems:

- The `audio_embedder` receives **exactly zero gradient on optimizer step 0** —
  AdaLN-zero makes the modulation weight exactly 0, so the audio path is literally
  disconnected at init and reconnects on step 1. So "audio does nothing" at epoch 0 is
  expected, not evidence.
- The N-way matching harness was **biased toward the correct answer**: with the true
  candidate always at position 0, `np.argmin`'s first-wins tie-breaking pushed a
  mean-image control to 3× chance. The true candidate is now placed at a random
  position, and the notebook runs a harness self-test (perfect reconstruction must be
  near ceiling, mean image must be at chance) before reporting anything.

## Statistics, briefly

- Decisive check null: `Binomial(300, 1/75)`, mean 4.000, SD 1.987. Critical values
  ≥8 (p<0.05), ≥10 (p<0.01), ≥12 (p<0.001).
- p-values use the **exact binomial survival function**. Do not use a normal tail on
  `(x−4)/1.987`: it is anti-conservative by 2.3× at 8 hits, 6.1× at 10, and 29× at 12.
- MDE at 80% power is 2.54× chance (~10 hits). Power is 25% at 1.5× chance, 55% at 2×,
  and `P(1 ≤ X ≤ 8) = 0.9616` under the null — a single-digit count is ordinary noise
  and does not establish "no effect". Report the CI, not just the verdict.

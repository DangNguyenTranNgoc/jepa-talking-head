# Diagnostics

Standalone checks produced during the 2026-08-27 external review. Neither needs
retraining; both run on checkpoints you already have.

## `tier0_audio_diagnostics.py`
Run BEFORE touching the architecture (~1 hour total).

1. `check_audio_features()`  — 5 min. `aud_ave.npy` is NOT an average: `ave` =
   Audio-Visual Encoder, Wav2Lip's LRS2-trained lip-sync tower over a 0.2 s mel
   window. It is a good feature; do not swap it. The problem is windowing —
   SyncTalk_2D feeds `features[i-8:i+8]` (16 vectors, 0.64 s); we feed 1.
2. `persistence_share()`    — 10 min. Decides whether audio is learnable at all.
3. `condition_driven_fraction()` — replaces the gate-magnitude check, which is
   useless: a zero-information condition drove gates to 0.306 while an
   informative one reached only 0.096. CDF < 0.02 = dead, > 0.3 = alive.
4. `held_out_shuffle_delta()` — the verdict. Run ONLY on a contiguous held-out
   block; a fixed random per-frame vector still scores +7% because it works as
   a frame ID the network memorises.

## `runE_diagnostic_cell.py`
Paste into `JEPA_Latent_Probing.ipynb` as Part 10, after Part 9d. ~2 minutes on
the epoch-11 checkpoint. Answers three things at once:

- shuffled-audio control  -> does the predictor use audio at all?
- persistence null        -> is it better than "hold still"?
- variance / cosine ratio -> has it partially collapsed or smoothed?

Full reasoning: `NNDT Knowledge Base/LeWM Talking-Head JEPA/` — see
`FaRL Verification and Decoder Plan` and `Next Steps Addendum v2`.

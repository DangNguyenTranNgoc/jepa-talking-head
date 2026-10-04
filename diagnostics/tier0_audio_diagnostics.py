"""
Tier 0 audio diagnostics — run before touching the architecture. ~1 hour total.

Supersedes the "log gate magnitude" advice from the previous report, which was
wrong: a condition carrying LITERALLY ZERO information drove gate magnitude to
0.306 while an informative one reached only 0.096. Gate magnitude measures
whether the transformer turned itself on (via the modulation BIAS), not whether
it uses audio.

The right measurement is the CONDITION-DRIVEN FRACTION: how much of the
modulation varies across different audio inputs, versus how much is a learned
constant.

Run order:
  1. check_audio_features()   -- 5 min, before any training
  2. persistence_share()      -- 10 min, decides whether audio is learnable at all
  3. condition_driven_fraction() -- on any trained checkpoint
  4. held_out_shuffle_delta() -- the verdict

ASSUMPTIONS: adjust names in each function's signature to your session.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. Feature sanity — do this FIRST, it is free
# --------------------------------------------------------------------------- #
def check_audio_features(aud_path="data/May/aud_ave.npy"):
    """aud_ave.npy is NOT an average.

    'ave' = Audio-Visual Encoder. Each 512-d row is Wav2Lip's LRS2-trained
    lip-sync expert encoding a 0.2 s mel window (SyncTalk nerf_triplane/utils.py:
    start_idx = int(80.*(frame/25.)); end_idx = start_idx + 16). It is a GOOD
    feature -- SyncTalk's own ablation has it beating HuBERT on LMD 2.8032 vs
    3.3961. Do not swap the encoder.

    The problem is windowing: SyncTalk_2D feeds features[i-8:i+8] = 16 vectors
    = 0.64 s. One vector = 0.2 s is a 16x reduction in temporal context.
    """
    c = np.load(aud_path)
    print(f"shape {c.shape}  (frames, dim)")

    mean, std = c.mean(0), c.std(0)
    ratio = np.linalg.norm(std) / (np.linalg.norm(mean) + 1e-9)
    print(f"||std||/||mean|| = {ratio:.4f}")
    if ratio < 0.5:
        print("  >>> Large constant offset relative to temporal variation.")
        print("      AdaLN's signal is W.(c - c_bar); a constant offset is")
        print("      structurally indistinguishable from the modulation bias.")
        print("      FIX (free): c = (c - c.mean(0)) / (c.std(0) + 1e-6)")

    # effective rank: a near-rank-1 condition cannot drive anything
    s = np.linalg.svd(c - mean, compute_uv=False)
    eff_rank = int((s > 0.01 * s.max()).sum())
    print(f"effective rank = {eff_rank} / {c.shape[1]}")
    if eff_rank < 10:
        print("  >>> Condition is near-degenerate. Nothing can be driven by this.")

    # lag-1 autocorrelation: how much NEW information arrives per frame
    cz = (c - mean) / (std + 1e-9)
    ac1 = float((cz[:-1] * cz[1:]).mean())
    print(f"lag-1 autocorrelation = {ac1:.4f}")
    print(f"  (high autocorrelation => consecutive frames carry little new signal)")
    return c


# --------------------------------------------------------------------------- #
# 2. Is audio learnable here at all?
# --------------------------------------------------------------------------- #
def persistence_share(emb):
    """emb: (T, D) real encoded embeddings for the whole clip.

    Wav2Lip found the lip region is "less than 4% of the total reconstruction
    loss" -- which is exactly why an L1/MSE objective lets a generator ignore
    audio. The same arithmetic applies to embedding-space MSE.

    If audio-attributable variance is a few percent of your loss, the copy
    solution wins and NO conditioning architecture will save you. Fix the
    objective instead (see predict_residual below).
    """
    e = torch.as_tensor(emb).float()
    mse_persist = ((e[1:] - e[:-1]) ** 2).mean().item()
    var_total = e.var(0).mean().item()
    unexplained = mse_persist / (var_total + 1e-9)
    print(f"persistence MSE      = {mse_persist:.6f}")
    print(f"total embedding var  = {var_total:.6f}")
    print(f"fraction NOT explained by persistence = {unexplained:.4f}")
    if unexplained < 0.05:
        print("  >>> Under 5%. Your loss is dominated by 'copy the last frame'.")
        print("      Predict the RESIDUAL e_t - e_{t-1} instead of e_t.")
    return unexplained


# --------------------------------------------------------------------------- #
# 3. The metric that replaces gate magnitude
# --------------------------------------------------------------------------- #
@torch.no_grad()
def condition_driven_fraction(model, cond_batch):
    """cond_batch: (B, cond_dim) -- B >= 32 DIFFERENT audio windows.

    CDF = cond_std / (cond_std + const_mag)

      < 0.02      DEAD   modulation is a learned constant
                         (the zero-information control sat at exactly 0.000)
      0.02 - 0.15 WEAK   bias-dominated; check the shuffle test
      0.3  - 0.8  ALIVE  the informative control held 0.66-0.74

    Report the GATE rows especially. Note that |mod| (magnitude) is printed only
    to show it is uninformative -- do not threshold on it.
    """
    names = ("shift1", "scale1", "gate1", "shift2", "scale2", "gate2")
    for bi, block in enumerate(model.predictor.transformer.layers):
        out = block.adaLN_modulation(cond_batch)          # (B, 6*D)
        print(f"-- block {bi} --")
        for name, t in zip(names, out.chunk(6, dim=1)):
            cond_std = t.std(0).mean().item()             # variation ACROSS conditions
            const_mag = t.mean(0).abs().mean().item()     # the bias-equivalent
            cdf = cond_std / (cond_std + const_mag + 1e-9)
            flag = ""
            if "gate" in name:
                flag = ("   <<< DEAD" if cdf < 0.02 else
                        "   <<< weak" if cdf < 0.15 else
                        "   <<< alive" if cdf > 0.30 else "")
            print(f"  {name:7s} |mod|={t.abs().mean():.4f}  "
                  f"cond_std={cond_std:.4f}  CDF={cdf:.3f}{flag}")


# --------------------------------------------------------------------------- #
# 4. The verdict — and the trap in it
# --------------------------------------------------------------------------- #
def held_out_shuffle_delta(loss_real, loss_shuffled, loss_zeroed=None):
    """Delta% = 100 * (L_shuffled - L_real) / L_real, on HELD-OUT data.

    THE TRAP: a fixed random vector per frame index -- carrying zero information
    about the target -- still reached +7% and CDF 0.34 in simulation, because
    the condition works as a unique FRAME ID the network memorises. On 6000
    frames of one speaker that is entirely feasible.

    So: hold out a CONTIGUOUS BLOCK, never random frames. At 25 fps a random
    held-out frame sits ~40 ms from a training frame and is effectively
    in-sample -- the same leakage that invalidated the FaRL comparison.

    Threshold: < 5% on contiguous held-out data means the audio pathway is not
    carrying audio.
    """
    d = 100.0 * (loss_shuffled - loss_real) / (loss_real + 1e-9)
    print(f"shuffled-audio delta = {d:+.1f}%")
    if loss_zeroed is not None:
        dz = 100.0 * (loss_zeroed - loss_real) / (loss_real + 1e-9)
        print(f"zeroed-audio   delta = {dz:+.1f}%")
    if d < 5:
        print("  >>> FAIL. The predictor is not using audio. Fix conditioning")
        print("      before any decoder or architecture work.")
    elif d < 20:
        print("  >>> WEAK. Some dependence. Report the magnitude honestly.")
    else:
        print("  >>> PASS. Audio carries load-bearing information.")
    return d


# --------------------------------------------------------------------------- #
# 5. The two Tier-0 fixes, for reference
# --------------------------------------------------------------------------- #
def whiten_audio(c, stats=None):
    """0a: free, and potentially the whole problem. Fit stats on TRAIN only."""
    if stats is None:
        stats = (c.mean(0, keepdims=True), c.std(0, keepdims=True) + 1e-6)
    mean, std = stats
    return (c - mean) / std, stats


def windowed_audio(features, index, radius=8):
    """Tier 1: match SyncTalk_2D exactly -- features[i-8:i+8] = 16 x 512 = 0.64 s.

    Reference (SyncTalk_2D/datasetsss.py):
        left = index - 8; right = index + 8
        auds = torch.from_numpy(features[left:right])
        ... audio_feat.reshape(32, 16, 16)      # 8192 = 16 frames x 512

    Consensus receptive field across AD-NeRF / RAD-NeRF / ER-NeRF / SyncTalk is
    0.5-0.85 s per predicted frame. One vector is 0.2 s.
    """
    left, right = index - radius, index + radius
    pad_l = max(0, -left)
    pad_r = max(0, right - len(features))
    w = features[max(0, left):min(len(features), right)]
    if pad_l or pad_r:
        w = np.pad(w, ((pad_l, pad_r), (0, 0)), mode="edge")
    return w                                   # (2*radius, dim)


def predict_residual_target(emb, ctx_len, num_preds):
    """0c: predict e_t - e_{t-1}, not e_t.

    Deletes the persistence shortcut from the objective and rescales the loss so
    audio-driven variance is a large fraction of the target rather than a few
    percent. Two lines here, one change at inference (cumulative sum from the
    last real context embedding). Ranked above most architecture changes.
    """
    tgt = emb[:, ctx_len:ctx_len + num_preds]
    prev = emb[:, ctx_len - 1:ctx_len + num_preds - 1]
    return tgt - prev


if __name__ == "__main__":
    print(__doc__)

"""
Run E diagnostic cell — paste into JEPA_Latent_Probing.ipynb as Part 10, after Part 9d.

Answers three questions in one ~2-minute pass on the already-loaded epoch-11
checkpoint. No retraining, no new landmarks, no new feature extraction.

  Q1  Does the predictor actually use audio?         -> shuffled-audio control
  Q2  Is it better than "hold still"?                -> persistence null
  Q3  Has it partially collapsed / smoothed?         -> variance + cosine diversity

Why each matters (see the verification report):
  Q1  ARPredictor's ConditionalBlock zero-initialises its AdaLN gates, so at
      init the model computes LayerNorm(ctx + pos_emb) -- a pure persistence
      predictor -- and audio enters ONLY through those gates. A run that stays
      near that regime reproduces every observed Run E number without using
      audio at all. There is currently no control anywhere that rules this out.
  Q2  The only null in use is mean-shape (4.99). For a 0.48 s rollout the
      relevant null is persistence. Run E ep11's pred barely varies across the
      12 steps (2.26 ... 2.57), which is exactly what "hold still" looks like.
  Q3  Nothing in either notebook measures the variance or diversity of any
      pred_emb tensor. A predictor that outputs a smoothed/low-pass trajectory
      scores near-zero NME gap while being useless for generation.

ASSUMPTIONS -- adjust the names in the CONFIG block to match your session.
Everything referenced should already be in memory after Part 9d.
"""

import numpy as np
import torch

# ----------------------------------------------------------------------------
# CONFIG - point these at your existing Part 8/9 objects
# ----------------------------------------------------------------------------
MODEL          = model              # Run E epoch-11, already loaded, .eval()
HISTORY_SIZE   = HISTORY_SIZE       # 4
NUM_PREDS      = NUM_PREDS          # 4
N_HOPS         = 3                  # 12 future frames
HOP3_STEPS     = HOP3_STEPS         # steps 9-12
PROBES         = chained_probes     # per-step fitted ridge probes from Part 8/9
Y_TEST         = Y_chained_norm     # shape-normalised landmark targets
TEST_IDX       = test_chained_idx
TRAIN_IDX      = train_chained_idx
BATCH          = chained_batch      # dict with 'pixels' and 'action'
SEED           = 1234


def _nme(pred_lm, true_lm):
    """Same convention as normalized_shape_error: centroid-removed, IOD-scaled."""
    return float(np.mean(np.linalg.norm(
        pred_lm.reshape(len(pred_lm), -1, 2) - true_lm.reshape(len(true_lm), -1, 2),
        axis=-1)) * 100)


def _score_steps(emb_by_step, idx, label):
    """Apply the already-fitted per-step probes to a set of embeddings."""
    out = {}
    for s, emb in emb_by_step.items():
        X = emb[idx].cpu().numpy().astype(np.float64)
        mean, std, ridge = PROBES[s]['mean'], PROBES[s]['std'], PROBES[s]['model']
        pred_lm = ridge.predict((X - mean) / std)
        out[s] = _nme(pred_lm, Y_TEST[s][idx])
    print(f'  {label:24s} ' + '  '.join(f'{out[s]:.2f}' for s in sorted(out)))
    return out


@torch.no_grad()
def chained_rollout(batch, shuffle_audio=False, seed=SEED):
    """Faithful re-implementation of Part 8's hand-built chaining.

    shuffle_audio=True permutes the audio across the BATCH dimension only,
    leaving every tensor shape, every code path, and the visual context
    untouched. The only thing destroyed is the audio<->video pairing.
    """
    out = MODEL.jepa.encode(batch)
    emb, act_emb = out['emb'], out['act_emb']

    if shuffle_audio:
        g = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(act_emb.size(0), generator=g).to(act_emb.device)
        # guard against identity permutation on small batches
        if torch.equal(perm, torch.arange(len(perm), device=perm.device)):
            perm = torch.roll(perm, 1)
        act_emb = act_emb[perm]

    ctx = emb[:, :HISTORY_SIZE]
    pred_by_step, real_by_step = {}, {}
    for hop in range(N_HOPS):
        a = HISTORY_SIZE + hop * NUM_PREDS
        b = a + NUM_PREDS
        pred_act = act_emb[:, a:b]                       # TARGET frames' audio
        if pred_act.shape[-1] != MODEL.embed_dim and hasattr(MODEL, 'safety_audio_proj'):
            pred_act = MODEL.safety_audio_proj(pred_act)
        pred_block = MODEL.jepa.predict(ctx, pred_act)
        for t in range(NUM_PREDS):
            step = hop * NUM_PREDS + t + 1
            pred_by_step[step] = pred_block[:, t]
            real_by_step[step] = emb[:, a + t]
        ctx = pred_block                                  # chain on own output
    return pred_by_step, real_by_step, emb


@torch.no_grad()
def persistence(emb, which='last'):
    """Null: broadcast one real context embedding across all 12 steps.

    'last'  -> ctx[:, -1]  the obvious persistence baseline
    'first' -> ctx[:,  0]  because causal masking means prediction position 0
                           attends only to context position 0, so this is the
                           frame the model may actually be copying
    """
    src = emb[:, HISTORY_SIZE - 1] if which == 'last' else emb[:, 0]
    return {s: src for s in range(1, N_HOPS * NUM_PREDS + 1)}


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
MODEL.eval()
steps = list(range(1, N_HOPS * NUM_PREDS + 1))
hdr = '  ' + ' ' * 24 + '  '.join(f'{s:>4d}' for s in steps)

print('=' * 78)
print('PART 10 - Run E ep11 diagnostics: audio dependence, persistence, collapse')
print('=' * 78)
print('\nNME % by step (lower better), held-out windows\n')
print(hdr)

pred_real, real_by_step, emb = chained_rollout(BATCH, shuffle_audio=False)
pred_shuf, _, _              = chained_rollout(BATCH, shuffle_audio=True)

r_real = _score_steps(real_by_step, TEST_IDX, 'real (ceiling)')
r_pred = _score_steps(pred_real,    TEST_IDX, 'pred (real audio)')
r_shuf = _score_steps(pred_shuf,    TEST_IDX, 'pred (SHUFFLED audio)')
r_pl   = _score_steps(persistence(emb, 'last'),  TEST_IDX, 'persistence (last ctx)')
r_pf   = _score_steps(persistence(emb, 'first'), TEST_IDX, 'persistence (first ctx)')

base = {}
for s in steps:
    m = Y_TEST[s][TRAIN_IDX].mean(0, keepdims=True)
    base[s] = _nme(np.repeat(m, len(TEST_IDX), 0), Y_TEST[s][TEST_IDX])
print(f'  {"mean-shape baseline":24s} ' + '  '.join(f'{base[s]:.2f}' for s in steps))

# --- Q1: audio dependence -----------------------------------------------------
d_hop3 = np.mean([r_shuf[s] - r_pred[s] for s in HOP3_STEPS])
d_all  = np.mean([r_shuf[s] - r_pred[s] for s in steps])
print('\n' + '-' * 78)
print('Q1  AUDIO DEPENDENCE  (shuffled - real; large positive = audio is used)')
print('-' * 78)
print(f'  all steps: {d_all:+.3f} NME     hop-3 (steps 9-12): {d_hop3:+.3f} NME')
if d_hop3 < 0.10:
    print('  >>> FAIL. Shuffling the audio barely changes the prediction.')
    print('      The predictor is doing motion extrapolation from the 4 context')
    print('      frames. "Audio-conditioned world model" is not supported, and the')
    print('      near-zero hop-3 gap is a persistence artifact the probe cannot see.')
elif d_hop3 < 0.50:
    print('  >>> WEAK. Some audio dependence, but most of the prediction is')
    print('      explained without it. Report the magnitude honestly.')
else:
    print('  >>> PASS. Audio carries real, load-bearing information.')
    print('      Remaining work is just re-reporting 0.11 +/- 0.04 with a CI.')

# --- Q2: persistence null -----------------------------------------------------
print('\n' + '-' * 78)
print('Q2  PERSISTENCE NULL  (does the predictor beat "hold still"?)')
print('-' * 78)
for name, r in (('last ctx frame', r_pl), ('first ctx frame', r_pf)):
    m_pl   = np.mean([r[s] for s in HOP3_STEPS])
    m_pred = np.mean([r_pred[s] for s in HOP3_STEPS])
    verdict = ('FAIL - not distinguishable from holding still' if m_pred > m_pl - 0.10
               else 'PASS')
    print(f'  vs {name:16s} hop-3: pred {m_pred:.2f} vs persistence {m_pl:.2f}'
          f'  -> {verdict}')

# --- Q3: collapse / smoothing -------------------------------------------------
print('\n' + '-' * 78)
print('Q3  PREDICTION DIVERSITY  (variance ratio < ~0.6 by hop 3 = smoothing)')
print('-' * 78)
print('  step   var_ratio   cos_real   cos_pred')
for s in steps:
    p, r = pred_real[s][TEST_IDX], real_by_step[s][TEST_IDX]
    vr = (p.std(0).mean() / r.std(0).mean().clamp_min(1e-8)).item()

    def _cos(x):
        xn = torch.nn.functional.normalize(x.float(), dim=-1)
        sim = xn @ xn.T
        n = len(sim)
        return ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

    flag = '  <-- LOW' if vr < 0.6 else ''
    print(f'  {s:4d}   {vr:9.3f}   {_cos(r):8.3f}   {_cos(p):8.3f}{flag}')
print('\n  cos_pred >> cos_real means predictions are collapsing toward each other,')
print('  i.e. the model outputs near-identical embeddings regardless of input.')

print('\n' + '=' * 78)
print('If Q1 fails, stop and fix audio conditioning before any decoder work.')
print('If Q1 passes, the next fix is the contiguous temporal split (see report).')
print('=' * 78)

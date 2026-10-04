"""Offline smoke test of S1's statistical core, on synthetic images."""
import json, sys, os, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2

nb = json.load(open('/home/claude/S1_Decoder_Oracle.ipynb'))
cells = ['\n'.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
def cell_with(m):
    h = [c for c in cells if m in c]
    assert h, f'no cell with {m!r}'
    return h[0]

IMG, PATCH, GRID, SEED = 224, 16, 14, 0
EMBED_DIM = 256
device = torch.device('cpu')
N_QUERIES, N_WAY = 300, 75

G = dict(np=np, torch=torch, nn=nn, F=F, cv2=cv2, os=os, json=json, math=math,
         sys=sys, IMG=IMG, PATCH=PATCH, GRID=GRID, SEED=SEED, device=device,
         EMBED_DIM=EMBED_DIM, N_QUERIES=N_QUERIES, N_WAY=N_WAY,
         tqdm=lambda x, **k: x, plt=None)

def run(m, label):
    exec(compile(cell_with(m), f'<cell:{label}>', 'exec'), G)
    print(f'  OK   {label}')

# --- synthetic clip: a fixed face + a moving "mouth" bar ----------------------
rng = np.random.default_rng(0)
N = 400
base = rng.uniform(0.3, 0.6, size=(IMG, IMG, 3)).astype(np.float32)
base = cv2.GaussianBlur(base, (0, 0), 9)
mouth_open = rng.uniform(0, 1, size=N).astype(np.float32)
frames = np.repeat(base[None], N, 0).copy()
# continuous, per-frame-unique mouth content: otherwise many frames are pixel-identical
# in the mouth crop, exact L2 ties abound, and even a perfect reconstruction cannot
# beat the tie-break. (Real 25 fps video does not have exact duplicates.)
for i in range(N):
    h = int(4 + 28 * mouth_open[i])
    tex = 0.05 + 0.85 * mouth_open[i] + rng.normal(0, 0.03, size=(h, 60, 3))
    frames[i, 150:150 + h, 90:150, :] = tex
PIXn = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
G['PIX'] = PIXn

TR_ROWS, TE_ROWS = np.arange(0, 300), np.arange(300, 400)
G.update(TR_ROWS=TR_ROWS, TE_ROWS=TE_ROWS, DENSE_TRAIN=TR_ROWS, TEST_IDX=TE_ROWS,
         POS={int(i): int(i) for i in range(N)}, L2_GRID=list(np.logspace(0, 6, 4)))

print('--- ridge helpers ---')
run('def ridge_fit', 'ridge helpers')

print('--- mouth box detection ---')
G['plt'] = __import__('matplotlib.pyplot', fromlist=['x'])
import matplotlib; matplotlib.use('Agg')
run('mouth box (top-left)', 'mouth box')
print(f'  detected MOUTH={G["MOUTH"]}  (truth: bar at rows 150.., cols 90..150)')
mr, mc = G['MOUTH']
assert 110 <= mr <= 175 and 60 <= mc <= 150, f'mouth box way off: {G["MOUTH"]}'

print('--- N-way matching harness ---')
run('def nway_match', 'nway_match')
nway_match = G['nway_match']

# (a) perfect reconstruction must be far above chance
perfect = PIXn[TE_ROWS].astype(np.float32) / 255.0
r_perfect = nway_match(perfect, TE_ROWS, n_queries=100, n_way=N_WAY)
print(f'  perfect recon      : {r_perfect["hits"]}/100  p={r_perfect["p_exact_binomial"]:.2e}')
assert r_perfect['hits'] > 80

# (b) the mean image must sit AT chance
meanimg = (PIXn[TR_ROWS].astype(np.float32) / 255.0).mean(0)
r_mean = nway_match(np.repeat(meanimg[None], len(TE_ROWS), 0), TE_ROWS, n_queries=100)
print(f'  mean-image control : {r_mean["hits"]}/100  expected {100/N_WAY:.2f}  '
      f'p={r_mean["p_exact_binomial"]:.3f}')
assert r_mean['hits'] <= 6, 'mean-image control is NOT at chance - harness is broken'

# (c) swapped latents must sit at chance
swap = perfect[np.random.default_rng(1).permutation(len(TE_ROWS))]
r_swap = nway_match(swap, TE_ROWS, n_queries=100)
print(f'  latent-swap control: {r_swap["hits"]}/100  p={r_swap["p_exact_binomial"]:.3f}')
assert r_swap['hits'] <= 6, 'swap control is not at chance'

# (d) exact binomial vs the anti-conservative normal approximation that was used before
from scipy.stats import binom, norm
print('\n  exact vs normal-tail p-values (the approximation that was used before):')
for k in (5, 8, 10, 12, 15):
    pe = binom.sf(k - 1, 300, 1 / 75)
    pn = norm.sf((k - 4.0) / 1.987)
    print(f'    {k:2d} hits   exact {pe:.4f}   normal-tail {pn:.4f}   '
          f'ratio {pe/pn:5.1f}x  {"<-- normal is anti-conservative" if pn < pe else ""}')

print('\n--- decoders forward/backward ---')
run('class VecDecoder', 'decoder classes')
vd = G['VecDecoder'](768)
out = vd(torch.randn(4, 768))
print(f'  VecDecoder(768)  -> {tuple(out.shape)}  range [{out.min():.2f}, {out.max():.2f}]')
assert out.shape == (4, 3, 224, 224)
dd = G['DenseDecoder']()
out2 = dd(torch.randn(2, 196, 768))
print(f'  DenseDecoder     -> {tuple(out2.shape)}')
assert out2.shape == (2, 3, 224, 224)
out2.mean().backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dd.parameters())
print('  gradients flow OK')

print('\n--- path-0 global ridge sanity (patch-mean surrogate) ---')
r2 = G['r2_vs_mean']
Xall = np.stack([PIXn[i].astype(np.float32).reshape(-1)[::200] for i in range(N)]) / 255.0
SMALL = 32
Ytr = np.stack([cv2.resize(PIXn[i], (SMALL, SMALL)).astype(np.float32).ravel() / 255. for i in TR_ROWS])
Yte = np.stack([cv2.resize(PIXn[i], (SMALL, SMALL)).astype(np.float32).ravel() / 255. for i in TE_ROWS])
mu, sd = Xall[TR_ROWS].mean(0, keepdims=True), Xall[TR_ROWS].std(0, keepdims=True) + 1e-6
W, b = G['ridge_fit']((Xall[TR_ROWS] - mu) / sd, Ytr, 10.0)
pred = G['ridge_predict']((Xall[TE_ROWS] - mu) / sd, W, b)
print(f'  informative features R2 vs mean-image: {r2(pred, Yte, Ytr.mean(0)):+.4f}  (want >> 0)')
ctrl = np.random.default_rng(9).normal(size=Xall.shape)
W2, b2 = G['ridge_fit'](ctrl[TR_ROWS], Ytr, 10.0)
print(f'  random-noise control R2 vs mean-image: '
      f'{r2(G["ridge_predict"](ctrl[TE_ROWS], W2, b2), Yte, Ytr.mean(0)):+.4f}  (want <= 0)')

print('\nALL S1 SMOKE CHECKS PASSED')

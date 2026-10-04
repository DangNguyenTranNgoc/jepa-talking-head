"""Offline smoke test of S0's load-bearing code, on synthetic data.
Executes the real notebook cells (not copies) so a bug here is a bug there."""
import json, sys, os, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, '/mnt/user-data/uploads/le-wm')
from module import MLP, ARPredictor

nb = json.load(open('/home/claude/S0_Instrument_And_Audio_Diagnostics.ipynb'))
cells = ['\n'.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

def cell_with(marker):
    hits = [c for c in cells if marker in c]
    assert len(hits) >= 1, f'no cell contains {marker!r}'
    return hits[0]

G = dict(globals())
G.update(dict(os=os, sys=sys, np=np, torch=torch, nn=nn, F=F, rearrange=rearrange,
              MLP=MLP, ARPredictor=ARPredictor,
              tqdm=lambda x, **k: x, math=__import__('math'), json=json,
              time=__import__('time'), gc=__import__('gc')))

# --- synthetic world ---------------------------------------------------------
N_FRAMES = 900
AUDIO_DIM, EMBED_DIM, HIDDEN_DIM, HISTORY_SIZE, NUM_PREDS = 512, 256, 256, 4, 4
N_HOPS_TRAIN = 3
SEQ_LEN = HISTORY_SIZE + N_HOPS_TRAIN * NUM_PREDS
SEED = 0
device = torch.device('cpu')
rng = np.random.default_rng(0)

# frames drift smoothly (persistence is strong) + an audio-driven component
t = np.arange(N_FRAMES)
audio = rng.normal(size=(N_FRAMES, AUDIO_DIM)).astype(np.float32) + 3.0   # big DC offset, like aud_ave
Wa = rng.normal(size=(AUDIO_DIM, 768)).astype(np.float32) * 0.02
drift = np.cumsum(rng.normal(size=(N_FRAMES, 768)).astype(np.float32) * 0.05, 0)
FARL = (drift + audio @ Wa).astype(np.float32)
AUD = audio
N_USABLE = N_FRAMES

TRAIN_IDX = np.arange(0, 800)
TEST_IDX = np.arange(800, 900)
FIT_IDX = TRAIN_IDX[:-100]
VAL_IDX = TRAIN_IDX[-100:]

os.makedirs('/tmp/fake_May', exist_ok=True)
np.save('/tmp/fake_May/aud_ave.npy', audio)
G['LOCAL_PATH'] = '/tmp/fake_May'
G['IDENTITY'] = 'May'

G.update(dict(N_FRAMES=N_FRAMES, AUDIO_DIM=AUDIO_DIM, EMBED_DIM=EMBED_DIM,
              HIDDEN_DIM=HIDDEN_DIM, HISTORY_SIZE=HISTORY_SIZE, NUM_PREDS=NUM_PREDS,
              N_HOPS_TRAIN=N_HOPS_TRAIN, SEQ_LEN=SEQ_LEN, SEED=SEED, device=device,
              FARL=FARL, AUD=AUD, N_USABLE=N_USABLE, TRAIN_IDX=TRAIN_IDX, TEST_IDX=TEST_IDX,
              FIT_IDX=FIT_IDX, VAL_IDX=VAL_IDX, BATCH_SIZE=8, LR=1e-4))

def run(marker, label):
    src = cell_with(marker)
    try:
        exec(compile(src, f'<cell:{label}>', 'exec'), G)
        print(f'  OK   {label}')
    except Exception as e:
        print(f'  FAIL {label}: {type(e).__name__}: {e}')
        raise

print('--- exec notebook cells ---')
run('def check_audio_features', 'audio sanity')          # needs AUD; harmless
run('def persistence_share', 'persistence share')
run('class AudioFeatureEmbedder', 'model defs')
run('block0 adaLN last layer', 'adaLN-zero check')
run('def make_windows', 'windows + gather_batch')
run('def scheduled_sampling_prob', 'scheduled-sampling loss')

print('\n--- shape / semantics checks ---')
model = G['model']
FIT_W, VAL_W, TEST_W = G['FIT_W'], G['VAL_W'], G['TEST_W']
assert FIT_W.max() + SEQ_LEN <= VAL_W.min() + 1
f, a = G['gather_batch'](FIT_W[:8], SEQ_LEN)
print(f'  gather_batch -> feats {tuple(f.shape)}, aud {tuple(a.shape)}')
assert f.shape == (8, SEQ_LEN, 768) and a.shape == (8, SEQ_LEN, AUDIO_DIM)

o = G['loss_scheduled'](model, f, a, 0, 100, train=True)
print(f'  loss_scheduled -> pred_loss {o["pred_loss"].item():.4f}, '
      f'pred_emb {tuple(o["pred_emb"].shape)}, hops {len(o["hop_losses"])}')
assert o['pred_emb'].shape == (8, N_HOPS_TRAIN * NUM_PREDS, EMBED_DIM)
assert o['tgt_emb'].shape == o['pred_emb'].shape

# audio alignment: hop h must be conditioned on audio of frames [ctx+h*np, ctx+(h+1)*np)
import inspect
src = cell_with('def scheduled_sampling_prob')
assert 'pred_act = act[:, start:end]' in src, 'audio alignment regression!'
print('  audio alignment: pred_act = act[:, start:end]  (future audio) OK')

# scheduled sampling ramp
sp = G['scheduled_sampling_prob']
assert sp(0, 100) == 0.0 and abs(sp(30, 100) - 0.5) < 1e-9 and sp(99, 100) == 0.5
print(f'  ramp: step0 {sp(0,100):.2f} -> step15 {sp(15,100):.2f} -> step30+ {sp(30,100):.2f} OK')

# gradient flow. At init AdaLN-zero makes the modulation weight exactly 0, so the
# audio_embedder receives EXACTLY zero gradient on step 0 - expected, not a bug. It
# reconnects on step 1, once adaLN_modulation's own weight has moved off zero.
o['pred_loss'].backward()
def grad_modules():
    return {n: any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in getattr(model, n).parameters())
            for n in ('projector', 'audio_embedder', 'predictor', 'pred_proj')}
g0 = grad_modules()
print(f'  step 0 gradient reach: {g0}')
assert g0['projector'] and g0['predictor'] and g0['pred_proj']
assert not g0['audio_embedder'], 'audio_embedder got gradient at init - AdaLN-zero broken?'

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
opt.step(); opt.zero_grad()
o2 = G['loss_scheduled'](model, f, a, 1, 100, train=True)
o2['pred_loss'].backward()
g1 = grad_modules()
print(f'  step 1 gradient reach: {g1}')
assert all(g1.values()), f'a module is still gradient-disconnected after one step: {g1}'
print('  AdaLN-zero: audio path connects on step 1, as designed OK')
opt.zero_grad()

print('\n--- CDF ---')
run('def condition_driven_fraction', 'CDF')
rows = G['cdf_rows']
gates = [r for r in rows if 'gate' in r['row']]
print(f'  {len(rows)} rows, {len(gates)} gate rows, 6 depth-6 blocks expected: '
      f'{len(rows)==36 and len(gates)==12}')
assert len(rows) == 36 and len(gates) == 12

print('\n--- rollout + bootstrap ---')
G['FIT_W'] = FIT_W
run('def rollout_eval', 'rollout_eval')
R = G['R']
for v in ('real', 'shuffled', 'zeroed'):
    assert len(R[v]['model']) == N_HOPS_TRAIN
    assert len(R[v]['model'][0]) == len(TEST_W), (len(R[v]['model'][0]), len(TEST_W))
print(f'  rollout over {len(TEST_W)} test windows x {N_HOPS_TRAIN} hops OK')
d = 100*(R['shuffled']['model'][0].mean()-R['real']['model'][0].mean())/R['real']['model'][0].mean()
print(f'  untrained-model shuffled delta hop1: {d:+.1f}% (expected ~0 at init: AdaLN-zero)')
lo, hi = G['boot_ci'](R['real']['model'][0], R['shuffled']['model'][0], n_boot=2000)
print(f'  bootstrap CI: [{lo:+.2f}, {hi:+.2f}]')

run('def report(R, hop', 'report')

print('\nALL S0 SMOKE CHECKS PASSED')

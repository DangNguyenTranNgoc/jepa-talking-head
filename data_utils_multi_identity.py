"""
data_utils_multi_identity.py

Standardizes new talking-head video clips into the exact data/<name>/ format
already used by data/May (gt_imgs/*.jpg + aud_ave.npy), so SyncTalkDataset can
train on multiple identities. Reuses SyncTalk's own preprocessing pipeline
(github.com/ZiqiaoPeng/SyncTalk) for frame extraction, face parsing, and
background/torso splitting, plus its bundled Audio Visual Encoder (AVE) for
audio features - rather than reimplementing any of that from scratch.

Deliberately SKIPS SyncTalk's NeRF-only pipeline stages: face landmarks, 3DMM
face-tracking (gated behind a manual Basel Face Model registration we don't
need), optical-flow bundle adjustment (the slowest stage by far), blendshape
capture, and camera transforms export. Our SyncTalkDataset only ever reads
gt_imgs/ and aud_ave.npy, so none of those stages are necessary.

Setup (run once, e.g. in a Colab cell):
    !pip install opencv-python scikit-learn scipy face-alignment librosa tqdm configargparse torchvision
(configargparse and torchvision are needed by SyncTalk's own face_parsing/test.py;
this script also auto-downloads its BiSeNet checkpoint on first run.)

Usage:
    python data_utils_multi_identity.py
        (downloads + preprocesses all 10 default identities into ./data/)

    python data_utils_multi_identity.py --identities CarlyFiorina ChuckGrassley
        (just a subset)

    python data_utils_multi_identity.py --skip-download
        (use if you've already placed data/<name>/<name>.<ext> yourself)

After this finishes, each data/<name>/ has gt_imgs/ + aud_ave.npy in the same
shape as data/May/. To train on all identities together, replace the single
SyncTalkDataset(data_root=...) call with:

    from torch.utils.data import ConcatDataset
    identity_dirs = ["data/May", "data/AnnMarieBuerkle", "data/AustinScott0", ...]
    full_dataset = ConcatDataset([SyncTalkDataset(data_root=d, audio_window_radius=5) for d in identity_dirs])
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SYNCTALK_REPO = "https://github.com/ZiqiaoPeng/SyncTalk.git"
SYNCTALK_DIR = Path("SyncTalk")  # cloned alongside wherever this script is run from

# 10 single-speaker HDTF clips from the mortal9826/HDTF_videos HF mirror -
# https://huggingface.co/datasets/mortal9826/HDTF_videos - chosen to be 10
# DISTINCT individuals (not multiple clips of the same person under
# different numeric suffixes), for maximum identity diversity per clip count.
DEFAULT_IDENTITIES = {
    "AnnMarieBuerkle": "AnnMarieBuerkle.mp4",
    "AustinScott0": "AustinScott0.mp4",
    "BillCassidy0": "BillCassidy0.mp4",
    "BobbySchilling": "BobbySchilling.mp4",
    "CandiceMiller": "CandiceMiller.mkv",
    "CarlyFiorina": "CarlyFiorina.webm",
    "ChuckGrassley": "ChuckGrassley.mkv",
    "DanNewhouse": "DanNewhouse.mkv",
    "DaveCamp": "DaveCamp.mp4",
    "DebFischer0": "DebFischer0.mkv",
}
HF_BASE_URL = "https://huggingface.co/datasets/mortal9826/HDTF_videos/resolve/main"


BISENET_CKPT_URL = "https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true"


def ensure_synctalk_repo():
    if not SYNCTALK_DIR.exists():
        print(f"[setup] cloning {SYNCTALK_REPO} ...")
        subprocess.run(["git", "clone", "--depth", "1", SYNCTALK_REPO, str(SYNCTALK_DIR)], check=True)
    else:
        print("[setup] SyncTalk repo already present, skipping clone")

    # SyncTalk's own README has this as a separate manual step - process.py's
    # face-parsing task assumes it's already there and fails silently (via
    # os.system, so the failure doesn't propagate) if it's missing.
    bisenet_ckpt = SYNCTALK_DIR / "data_utils" / "face_parsing" / "79999_iter.pth"
    if not bisenet_ckpt.exists():
        print(f"[setup] downloading BiSeNet face-parsing checkpoint -> {bisenet_ckpt}")
        subprocess.run(["wget", "-q", "-O", str(bisenet_ckpt), BISENET_CKPT_URL], check=True)

    # configargparse/torchvision are needed by data_utils/face_parsing/test.py
    # but aren't in a typical Colab image; installing here too in case the
    # notebook cell's !pip install line was skipped or ran before a runtime
    # restart wiped it.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "configargparse", "torchvision"], check=True)

    patch_face_parsing_test()
    patch_extract_background()


_FACE_PARSING_PATCH_MARKER = "# PATCHED (data_utils_multi_identity.py): tolerate zero-face frames"

# The exact block from SyncTalk's data_utils/face_parsing/test.py's
# vis_parsing_maps(), verified against the upstream source. It computes a
# mouth-region debug overlay and writes it to "<n>_face.png" - a file NEVER
# read by extract_background()/extract_torso_and_gt() in process.py (they
# only read the plain "<n>.png" written earlier in the SAME function, before
# this block runs). When a frame has zero face-class pixels (e.g. an
# intro/cutaway frame with no face on screen), face_coords ends up empty and
# np.min() on it raises ValueError - which, because evaluate()'s for-loop has
# no per-frame try/except, kills the ENTIRE face-parsing run, silently
# (os.system() in process.py swallows the crash), leaving every frame after
# the crash point without a parsing/*.png file at all. Since the crashing
# code's own output isn't used downstream, the safe fix is to just skip it
# for that frame instead of aborting the whole batch.
_ORIGINAL_FACE_BLOCK = '''    pad = 5
    vis_parsing_anno_color_face = vis_parsing_anno_color_face.astype(np.uint8)
    face_part = (vis_parsing_anno_color_face[..., 0] == 255) & (vis_parsing_anno_color_face[..., 1] == 0) & (vis_parsing_anno_color_face[..., 2] == 0)
    face_coords = np.stack(np.nonzero(face_part), axis=-1)
    sorted_inds = np.lexsort((-face_coords[:, 0], face_coords[:, 1]))
    sorted_face_coords = face_coords[sorted_inds]
    u, uid, ucnt = np.unique(sorted_face_coords[:, 1], return_index=True, return_counts=True)
    bottom_face_coords = sorted_face_coords[uid] + np.array([pad, 0])
    rows, cols, _ = vis_parsing_anno_color_face.shape

    # 为了保证新的坐标在图片范围内
    bottom_face_coords[:, 0] = np.clip(bottom_face_coords[:, 0], 0, rows - 1)

    y_min = np.min(bottom_face_coords[:, 1])
    y_max = np.max(bottom_face_coords[:, 1])

    # 计算1和2部分的开始和结束位置
    y_range = y_max - y_min
    height_per_part = y_range // 4

    start_y_part1 = y_min + height_per_part
    end_y_part1 = start_y_part1 + height_per_part

    start_y_part2 = end_y_part1
    end_y_part2 = start_y_part2 + height_per_part

    for coord in bottom_face_coords:
        x, y = coord
        start_x = max(x - pad, 0)
        end_x = min(x + pad, rows)
        if start_y_part1 <= y <= end_y_part1 or start_y_part2 <= y <= end_y_part2:
            vis_parsing_anno_color_face[start_x:end_x, y] = [255, 0, 0]
        # else:
        #     start_x = max(x - 2*pad, 0)
        #     end_x = max(x - pad, 0)
        #     vis_parsing_anno_color_face[start_x:end_x+1, y] = [255, 255, 255]

    vis_im = cv2.GaussianBlur(vis_parsing_anno_color_face, (9, 9), cv2.BORDER_DEFAULT)

    vis_im = cv2.resize(vis_im, img_size,
                        interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_path.replace('.png', '_face.png'), vis_im)'''

_PATCHED_FACE_BLOCK = f'''    {_FACE_PARSING_PATCH_MARKER}
    pad = 5
    vis_parsing_anno_color_face = vis_parsing_anno_color_face.astype(np.uint8)
    face_part = (vis_parsing_anno_color_face[..., 0] == 255) & (vis_parsing_anno_color_face[..., 1] == 0) & (vis_parsing_anno_color_face[..., 2] == 0)
    try:
        face_coords = np.stack(np.nonzero(face_part), axis=-1)
        if face_coords.shape[0] == 0:
            raise ValueError("no face-class pixels detected in this frame")
        sorted_inds = np.lexsort((-face_coords[:, 0], face_coords[:, 1]))
        sorted_face_coords = face_coords[sorted_inds]
        u, uid, ucnt = np.unique(sorted_face_coords[:, 1], return_index=True, return_counts=True)
        bottom_face_coords = sorted_face_coords[uid] + np.array([pad, 0])
        rows, cols, _ = vis_parsing_anno_color_face.shape
        bottom_face_coords[:, 0] = np.clip(bottom_face_coords[:, 0], 0, rows - 1)

        y_min = np.min(bottom_face_coords[:, 1])
        y_max = np.max(bottom_face_coords[:, 1])
        y_range = y_max - y_min
        height_per_part = y_range // 4

        start_y_part1 = y_min + height_per_part
        end_y_part1 = start_y_part1 + height_per_part
        start_y_part2 = end_y_part1
        end_y_part2 = start_y_part2 + height_per_part

        for coord in bottom_face_coords:
            x, y = coord
            start_x = max(x - pad, 0)
            end_x = min(x + pad, rows)
            if start_y_part1 <= y <= end_y_part1 or start_y_part2 <= y <= end_y_part2:
                vis_parsing_anno_color_face[start_x:end_x, y] = [255, 0, 0]

        vis_im = cv2.GaussianBlur(vis_parsing_anno_color_face, (9, 9), cv2.BORDER_DEFAULT)
        vis_im = cv2.resize(vis_im, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path.replace('.png', '_face.png'), vis_im)
    except (ValueError, IndexError) as e:
        # This frame's BiSeNet output has no face-class pixels at all - the
        # ORIGINAL code crashed the whole evaluate() loop here, silently
        # killing every frame after it. "<n>_face.png" (this block's only
        # output) is never read downstream, so it's safe to just skip it.
        print(f"[patched] skipping _face.png overlay for {{save_path}}: {{e}}")'''


_BG_PATCH_MARKER = "# PATCHED (data_utils_multi_identity.py): drop zero-foreground sampled frames"

# The exact block from SyncTalk's data_utils/process.py's extract_background(),
# verified against the upstream source. extract_background samples every 20th
# ori_imgs frame (image_paths[::20]) and, for each, fits a NearestNeighbors on
# that frame's foreground pixels (fg_xys) to build a background plate. If a
# sampled frame's parsing map has ZERO foreground pixels (e.g. an intro/cutaway
# frame with no person on screen at all - the same class of frame the
# face_parsing patch above tolerates), fg_xys is empty and .fit() raises
# ValueError, crashing task 4 for real (this one does propagate, unlike task 3's
# os.system()-swallowed crash). A fully blank sampled frame doesn't help this
# background-plate estimator anyway (there's no visible background to sample
# around a subject), so it's safe to just drop it from the sample instead of
# crashing the whole task, as long as at least one sampled frame has some
# foreground.
_ORIGINAL_BG_BLOCK = '''    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)'''

_PATCHED_BG_BLOCK = f'''    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]

    {_BG_PATCH_MARKER}
    filtered_image_paths = []
    for image_path in image_paths:
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        if (~bg).any():
            filtered_image_paths.append(image_path)
        else:
            print(f"[patched] dropping {{image_path}} from background sampling: zero foreground pixels")
    if len(filtered_image_paths) == 0:
        raise RuntimeError("extract_background: every sampled frame has zero foreground pixels")
    image_paths = filtered_image_paths

    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)'''


def _apply_source_patch(path: Path, marker: str, original_block: str, patched_block: str, label: str):
    """Idempotently applies an exact-string-match patch to a vendored/cloned
    third-party file, skipping if already applied and warning (rather than
    silently doing nothing useful) if the expected block isn't found."""
    src = path.read_text()
    if marker in src:
        print(f"[setup] {label} already patched, skipping")
        return
    if original_block not in src:
        print(
            f"[setup] WARNING: could not find the expected block in {path} to patch "
            f"({label}) - upstream file may have changed. Leaving it untouched."
        )
        return
    path.write_text(src.replace(original_block, patched_block))
    print(f"[setup] patched {path} ({label})")


def patch_face_parsing_test():
    """See _ORIGINAL_FACE_BLOCK's comment above for why this is safe."""
    _apply_source_patch(
        SYNCTALK_DIR / "data_utils" / "face_parsing" / "test.py",
        _FACE_PARSING_PATCH_MARKER, _ORIGINAL_FACE_BLOCK, _PATCHED_FACE_BLOCK,
        "tolerate zero-face frames",
    )


def patch_extract_background():
    """See _ORIGINAL_BG_BLOCK's comment above for why this is safe."""
    _apply_source_patch(
        SYNCTALK_DIR / "data_utils" / "process.py",
        _BG_PATCH_MARKER, _ORIGINAL_BG_BLOCK, _PATCHED_BG_BLOCK,
        "drop zero-foreground sampled frames",
    )


def download_identity_video(name, filename, data_root):
    dest_dir = data_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{name}{Path(filename).suffix}"
    if dest_path.exists():
        print(f"[download] {dest_path} already exists, skipping")
        return dest_path
    url = f"{HF_BASE_URL}/{filename}"
    print(f"[download] {url} -> {dest_path}")
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest_path), url], check=True)
    return dest_path


def compute_face_crop_box(video_path: Path, num_samples: int = 5, margin_scale: float = 3.2):
    """Samples a few frames and finds a face-centered square crop box in
    ORIGINAL-video pixel coordinates. Root cause of the zero-size-array /
    NearestNeighbors crashes below: the HDTF mirror we download from hosts
    the raw, uncropped interview footage (e.g. 1280x720, face occupying a
    small fraction of frame), not HDTF's own curated 512x512 face crops.
    SyncTalk's face_parsing/BiSeNet was tuned on tightly-cropped faces (like
    data/May), so on a wide shot it can classify a frame as having ZERO
    face-class pixels, which crashes vis_parsing_maps -> then crashes
    extract_background's NearestNeighbors on that frame's empty foreground.
    Cropping to a face-centered square before running SyncTalk's pipeline
    fixes this at the source instead of patching around it."""
    import cv2
    import face_alignment

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_idxs = np.linspace(0, max(total - 1, 0), num=num_samples, dtype=int)

    # face_alignment renamed LandmarksType._2D -> LandmarksType.TWO_D between
    # versions; try both, same as SyncTalk's own extract_landmarks does.
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except AttributeError:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    boxes = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)
        if not preds:
            continue
        lms = preds[0]
        boxes.append((lms[:, 0].min(), lms[:, 1].min(), lms[:, 0].max(), lms[:, 1].max()))
    cap.release()
    del fa

    if not boxes:
        print("[crop] no face detected on any sampled frame, falling back to a center square crop")
        side = min(frame_w, frame_h)
        x = (frame_w - side) // 2
        y = (frame_h - side) // 2
        return x, y, side, side

    boxes = np.array(boxes)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    face_h = y1 - y0
    side = min(face_h * margin_scale, frame_w, frame_h)
    x = int(np.clip(cx - side / 2, 0, frame_w - side))
    y = int(np.clip(cy - side / 2, 0, frame_h - side))
    return x, y, int(side), int(side)


def crop_and_scale_video(video_path: Path, out_path: Path, box, size: int = 512):
    x, y, w, h = box
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"crop={w}:{h}:{x}:{y},scale={size}:{size}",
        "-c:a", "copy",
        str(out_path),
    ]
    print(f"[crop] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_face_cropped_video(name: str, video_path: Path) -> Path:
    """Returns a face-centered, 512x512 version of video_path, creating it
    (alongside the original, same data/<name>/ folder) if needed. See
    compute_face_crop_box for why this is necessary."""
    cropped_path = video_path.with_name(f"{name}_512.mp4")
    if cropped_path.exists():
        print(f"[crop] {cropped_path} already exists, skipping")
        return cropped_path
    box = compute_face_crop_box(video_path)
    print(f"[crop] {name}: detected face crop box (x,y,w,h)={box}")
    crop_and_scale_video(video_path, cropped_path, box)
    return cropped_path


def run_process_task(video_path: Path, task: int):
    """Runs SyncTalk's data_utils/process.py for a single task number. cwd is
    set to the SyncTalk clone (its own subprocess calls use paths relative to
    its repo root), but the video path passed in is ABSOLUTE, so process.py's
    outputs land in our own project's data/<name>/ folder, not inside the
    SyncTalk clone itself."""
    cmd = [sys.executable, "data_utils/process.py", str(video_path.resolve()), "--task", str(task)]
    print(f"[process.py] task {task}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SYNCTALK_DIR), check=True)


def preprocess_identity(name, video_path: Path):
    """Tasks 1-5 only, per SyncTalk's data_utils/process.py --task flag:
    1=extract_audio, 2=extract_images, 3=extract_semantics (BiSeNet face
    parsing), 4=extract_background, 5=extract_torso_and_gt (produces
    gt_imgs/). Tasks 6-10 (landmarks, 3DMM face tracking, optical flow,
    blendshapes, NeRF transforms) are skipped entirely - SyncTalkDataset
    never reads their outputs."""
    print(f"\n=== preprocessing {name} ({video_path.name}) ===")
    base_dir = video_path.parent

    def count_parsing_pngs():
        # vis_parsing_maps writes TWO files per frame when it can: "<n>.png"
        # (the actual segmentation map extract_background/extract_torso_and_gt
        # read) and "<n>_face.png" (a debug mouth-overlay, unused downstream,
        # skipped by our patch on zero-face frames). Only count the former -
        # glob("*.png") alone double-counts frames that got both.
        return len([p for p in (base_dir / "parsing").glob("*.png") if not p.stem.endswith("_face")])

    n_ori_existing = len(list((base_dir / "ori_imgs").glob("*.jpg")))
    if n_ori_existing > 0:
        print(f"[process.py] {name}: ori_imgs/ already has {n_ori_existing} frames, skipping tasks 1-2")
    else:
        run_process_task(video_path, 1)
        run_process_task(video_path, 2)
    n_ori = len(list((base_dir / "ori_imgs").glob("*.jpg")))
    if n_ori == 0:
        raise RuntimeError(f"[{name}] task 2 (extract_images) produced no frames in ori_imgs/")

    # task 3 shells out via os.system() internally, so a failure inside it
    # (e.g. a missing dependency, or - before the vis_parsing_maps patch - a
    # frame with zero face-class pixels) does NOT raise here - verify its
    # actual output directly rather than trusting the exit code.
    if count_parsing_pngs() == n_ori:
        print(f"[process.py] {name}: parsing/ already complete ({n_ori} frames), skipping task 3")
    else:
        run_process_task(video_path, 3)
    n_parsing = count_parsing_pngs()
    if n_parsing != n_ori:
        raise RuntimeError(
            f"[{name}] task 3 (extract_semantics/BiSeNet parsing) produced {n_parsing}/{n_ori} "
            f"frames in parsing/ - check the printed traceback above."
        )

    if (base_dir / "bc.jpg").exists():
        print(f"[process.py] {name}: bc.jpg already exists, skipping task 4")
    else:
        run_process_task(video_path, 4)
    if not (base_dir / "bc.jpg").exists():
        raise RuntimeError(f"[{name}] task 4 (extract_background) did not produce bc.jpg")

    gt_dir = base_dir / "gt_imgs"
    if len(list(gt_dir.glob("*.jpg"))) >= n_ori:
        print(f"[process.py] {name}: gt_imgs/ already complete, skipping task 5")
    else:
        run_process_task(video_path, 5)
    n_frames = len(list(gt_dir.glob("*.jpg")))
    if n_frames == 0:
        raise RuntimeError(f"[{name}] task 5 (extract_torso_and_gt) produced no output in gt_imgs/")
    print(f"[process.py] {name}: {n_frames} gt_imgs frames produced")


# --- Vendored verbatim from SyncTalk's nerf_triplane/network.py (AudioEncoder) and
# nerf_triplane/utils.py (melspectrogram/AudDataset), rather than importing those
# modules directly - nerf_triplane/network.py also pulls in nerf_triplane/renderer.py
# via `from .renderer import NeRFRenderer`, which does `import raymarching` at module
# level (a compiled CUDA extension only needed for actual NeRF rendering/training,
# requiring a separate `pip install ./raymarching` build step). None of that is
# needed just to run the audio encoder, so these are copied standalone instead of
# importing the real package, to avoid an unrelated and unnecessary CUDA build step.
import torch.nn as nn
import librosa
from scipy import signal as _scipy_signal


class _Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncTalkAudioEncoder(nn.Module):
    """Verbatim copy of SyncTalk's AudioEncoder (nerf_triplane/network.py) -
    the "AVE" (Audio Visual Encoder). Outputs a 512-dim feature per mel window."""

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            _Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            _Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            _Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.audio_encoder(x)
        return out.squeeze(2).squeeze(2)


# --- Vendored verbatim from SyncTalk's nerf_triplane/utils.py (melspectrogram +
# AudDataset) - same Wav2Lip-style mel-spectrogram constants SyncTalk itself uses.
def _load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def _preemphasis(wav, k):
    return _scipy_signal.lfilter([1, -k], [1], wav)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _linear_to_mel(spectrogram):
    return np.dot(_build_mel_basis(), spectrogram)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.0) * ((S - -100) / 100) - 4.0, -4.0, 4.0)


def _melspectrogram(wav):
    D = _stft(_preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    return _normalize(S)


class SyncTalkAudDataset(torch.utils.data.Dataset):
    """Verbatim copy of SyncTalk's AudDataset (nerf_triplane/utils.py)."""

    def __init__(self, wavpath):
        wav = _load_wav(wavpath, 16000)
        self.orig_mel = _melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80.0 * 25.0) + 2

    def _crop_audio_window(self, spec, start_frame):
        start_idx = int(80.0 * (start_frame / 25.0))
        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            end_idx = spec.shape[0]
            start_idx = end_idx - 16
        return spec[start_idx:end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        mel = self._crop_audio_window(self.orig_mel.copy(), idx)
        if mel.shape[0] != 16:
            raise RuntimeError("mel.shape[0] != 16")
        return torch.FloatTensor(mel.T).unsqueeze(0)


def extract_ave_features(name, video_path: Path):
    """Reproduces exactly what SyncTalk's own nerf_triplane/provider.py does
    for asr_model='ave' (process.py's task 1 deliberately skips this step -
    its extract_audio_features() prints 'AVE has been integrated into the
    training code' and returns - so aud_ave.npy has to be produced this way,
    not by process.py). Matches data/May/aud_ave.npy's format exactly:
    512-dim per video frame, first/last frame each repeated x2 to align
    counts with the video's frame count. Uses the vendored classes above
    instead of importing SyncTalk's nerf_triplane package (see comment above)."""
    base_dir = video_path.parent
    wav_path = base_dir / "aud.wav"
    out_path = base_dir / "aud_ave.npy"
    if out_path.exists():
        print(f"[ave] {out_path} already exists, skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncTalkAudioEncoder().to(device).eval()
    ckpt_path = SYNCTALK_DIR / "nerf_triplane" / "checkpoints" / "audio_visual_encoder.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})

    dataset = SyncTalkAudDataset(str(wav_path))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs = []
    with torch.no_grad():
        for mel in loader:
            outputs.append(model(mel.to(device)))
    outputs = torch.cat(outputs, dim=0).cpu()
    # pad 2 frames at each edge to align with the video's frame count, exactly
    # as nerf_triplane/provider.py does for the 'ave' asr model.
    first, last = outputs[:1], outputs[-1:]
    aud_features = torch.cat([first.repeat(2, 1), outputs, last.repeat(2, 1)], dim=0).numpy()
    np.save(out_path, aud_features)
    print(f"[ave] {name}: saved {out_path} shape={aud_features.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data",
                         help="project data/ folder (the same one containing May/)")
    parser.add_argument("--identities", nargs="*", default=None,
                         help="subset of DEFAULT_IDENTITIES names to process; default is all 10")
    parser.add_argument("--skip-download", action="store_true",
                         help="use this if you've already placed data/<name>/<name>.<ext> yourself")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    ensure_synctalk_repo()

    names = args.identities or list(DEFAULT_IDENTITIES.keys())
    for name in names:
        if name not in DEFAULT_IDENTITIES:
            print(f"[skip] unknown identity '{name}', not in DEFAULT_IDENTITIES")
            continue
        filename = DEFAULT_IDENTITIES[name]
        if args.skip_download:
            matches = list((data_root / name).glob(f"{name}.*"))
            if not matches:
                print(f"[skip] --skip-download set but no {data_root / name}/{name}.* found")
                continue
            video_path = matches[0]
        else:
            video_path = download_identity_video(name, filename, data_root)

        video_path = ensure_face_cropped_video(name, video_path)
        preprocess_identity(name, video_path)
        extract_ave_features(name, video_path)

    print("\nDone. Each data/<name>/ now has gt_imgs/ and aud_ave.npy, matching data/May/'s format.")
    print("To train on all identities together, replace SyncTalkDataset(data_root=...) with:")
    print("  from torch.utils.data import ConcatDataset")
    print("  full_dataset = ConcatDataset([SyncTalkDataset(data_root=d, audio_window_radius=5) for d in identity_dirs])")


if __name__ == "__main__":
    main()

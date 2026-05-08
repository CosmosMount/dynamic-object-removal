# Video Object Removal (clean-slate refactor)

This repository is a **pure Python** pipeline with explicit stages:

**mask → track → inpaint → eval**

All code lives under `src/object_removal/`. Legacy `pipelines/`/`scripts/` have been removed.

## Repository structure (new)
- `src/object_removal/`
  - `cli/`: per-stage CLIs + `compare`
  - `stages/`: stage orchestrators (stable I/O contracts)
  - `methods/`: interchangeable implementations (sam2/sam3/propainter/diffueraser/trackanything/…)
  - `io/`: canonical run directory layout + frame/mask helpers
- `modules/`: third-party modules (VGGT4D, SAM2, SAM3, ProPainter, DiffuEraser, Track-Anything, …)
- `ckpts/`: model checkpoints / weights (not tracked by git; see `.gitignore`)
- `data/DAVIS/`: DAVIS dataset (frames + annotations)

### Checkpoints layout (`ckpts/`)

Place downloaded weights here (repo code resolves these paths by default):

- `ckpts/sam2/*.pt` — SAM2 checkpoints (e.g. `sam2.1_hiera_large.pt`)
- `ckpts/sam3/sam3.pt` — SAM3 video checkpoint (+ tokenizer files if you keep them alongside)
- `ckpts/trackanything/` — Segment Anything + XMem weights for Track-Anything
- `ckpts/yolo/yolov8n-seg.pt` — YOLOv8-seg for `mask=yolo_first`, `track=optflow`, Track-Anything init, baseline YOLO branch
- `ckpts/propainter/weights/` — ProPainter auto-download cache (RAFT / completion / `ProPainter.pth`; created when you run `inpaint=propainter`)
- `ckpts/diffueraser/weights/` — DiffuEraser bundle (`stable-diffusion-v1-5`, `sd-vae-ft-mse`, `diffuEraser`, `propainter`, `PCM_Weights`, …)

### Evaluation (`cli/eval`)

`eval` 在 `src/object_removal/stages/eval_stage.py` 内用 **OpenCV + NumPy** 计算 mask IoU（J&F 风格）与帧 PSNR/SSIM；**不需要** 安装或保留 `davis2017-evaluation` 仓库。若你手头有官方 DAVIS 评测脚本生成的指标 CSV，可通过 `--davis_csv` 读入其中的 J-Mean / J-Recall 字段（可选）。

### `modules/` 体积说明

为只保留跑通本仓库流水线所需内容，已删除各子项目中的训练脚本、示例数据、演示与评测辅助目录等；推理仍依赖各自的 `sam2/`、`sam3/`、`model/`、`diffueraser/` 等代码包，请勿再手动删掉这些目录。

## Running (recommended)

We run modules in their own conda envs (the code does **not** try to switch envs automatically).

### Conda environments (per stage / method)

The rule is simple:

- **The CLIs run in the currently activated env.** If a pipeline uses multiple heavy modules with conflicting deps, you must **run stages separately** under the corresponding envs.

Conda env names (as used in this repo/docs):

- **vggt**: VGGT4D mask generation
  - Used by: `mask=vggt4d|vggt_framewise`
  - Needs: `modules/VGGT4D/requirements.txt` (notably `pycolmap`, `open3d`, `onnxruntime`, …)
- **sam2**: SAM2 tracker (YOLO init is also commonly run in this env)
  - Used by: `track=sam2` and often `mask=yolo_first`
  - Needs: `modules/sam2` deps (see `modules/sam2/pyproject.toml`)
- **sam3**: SAM3 tracker
  - Used by: `track=sam3`
  - Needs: `modules/sam3` deps (see `modules/sam3/pyproject.toml`)
- **propainter**: ProPainter inpainting
  - Used by: `inpaint=propainter`
  - Needs: `modules/ProPainter/requirements.txt`
- **diffueraser**: DiffuEraser inpainting
  - Used by: `inpaint=diffueraser`
  - Needs: `modules/DiffuEraser/requirements.txt` (pins `torch==2.3.1` / `diffusers==0.29.2` etc)
  - Install **`ffmpeg`** (e.g. `/usr/bin/ffmpeg`): outputs are re-encoded to H.264 (yuv420p + faststart) so `diffueraser_result.mp4` plays in VS Code. `compare`’s `conda run` prepends `/usr/bin` to `PATH` so re-encode can find it; if re-encode still fails, check the stderr line printed by the inpaint step.
- **trackanything**: Track-Anything tracker
  - Used by: `track=trackanything`
  - Needs: `modules/Track-Anything/requirements.txt` (+ checkpoints under `ckpts/trackanything/`)
- **davis**: evaluation / lightweight utilities
  - Used by: `eval` + `track=identity|optflow` + `inpaint=baseline_handcrafted`
  - Python deps: `numpy`, `opencv-python`

### When to activate which env

Because we do **not** auto-switch envs, you have two safe ways to run:

- **One env run**: only for pipelines whose `mask/track/inpaint/eval` all work in the *same* env.
- **Stage-by-stage**: activate the right env for each stage, reusing the same `run_dir`.

Examples (stage-by-stage):

- **baseline** (all in `davis`):
  - `conda activate davis` → run `mask` (baseline/yolo) → `track` (identity/optflow) → `inpaint` (handcrafted) → `eval`
- **yolosam2** (common split):
  - `conda activate sam2` → run `mask=yolo_first` + `track=sam2`
  - `conda activate propainter` → run `inpaint=propainter`
  - `conda activate davis` → run `eval`
- **vggt4dsam3_diffueraser** (common split):
  - `conda activate vggt` → run `mask=vggt4d`
  - `conda activate sam3` → run `track=sam3`
  - `conda activate diffueraser` → run `inpaint=diffueraser`
  - `conda activate davis` → run `eval`
- **vggt_trackanything_diffueraser** (common split):
  - `conda activate vggt` → `mask=vggt4d`
  - `conda activate trackanything` → `track=trackanything`
  - `conda activate diffueraser` → `inpaint=diffueraser`
  - `conda activate davis` → `eval`

All commands below assume you are at repo root.

Recommended (per conda env): install this repo as editable so you don't need `PYTHONPATH`:

```bash
pip install -e .
```

### One-command comparison on a DAVIS sequence

Run multiple pipelines on the same sequence and aggregate metrics:

By default, `compare` will **auto-run each stage in its corresponding conda env** via `conda run`.
This means you can run one command from a lightweight core env, as long as the target envs exist
and each env has this repo installed (`pip install -e .`).

```bash
python -m object_removal.cli.compare \
  --task davis:bmx-trees \
  --davis_root data/DAVIS \
  --pipelines baseline yolosam2 yoloopt \
  --out_root outputs/compare/bmx-trees
```

Notes:
- `baseline`: handcrafted baseline (single env with numpy/opencv; YOLO optional)
- `yolosam2`: YOLO init → SAM2 track → ProPainter inpaint
- `yoloopt`: YOLO init → optflow track → ProPainter inpaint
- Env mapping is configured in `configs/env_map.json` (method → conda env). Override with `--env_map` if needed.
  Empty string means “run in the same env as `compare`” for that stage (you must have that stage’s dependencies installed there).
- Use `--env_policy force_single` to disable auto env switching (runs everything in current env).

### Run stages manually (per-stage CLI)

Mask:

```bash
python -m object_removal.cli.mask --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method yolo_first
```

Track:

```bash
python -m object_removal.cli.track --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam2
```

Inpaint:

```bash
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method propainter
```

Eval:

```bash
python -m object_removal.cli.eval --run_dir runs/demo \
  --pred_mask_dir runs/demo/track/masks_binary \
  --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees \
  --pred_frames_dir runs/demo/inpaint/frames \
  --gt_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```

## Work log

See `TASK.md` for a detailed refactor log.

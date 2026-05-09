# Dynamic Object Removal — Deployment & End-to-End Guide

This repository implements a modular video object removal pipeline split into four explicit stages:

**mask → track → inpaint → eval**

- **Core pipeline code**: `src/object_removal/`
- **Vendored third-party methods**: `modules/`
- **Run configuration**: `configs/compare.yaml` (one run) and `configs/pipelines.yaml` (pipelines + params)

Chinese version: **`README_zh.md`**

---

## 0. Quick start (recommended)

1. Prepare DAVIS under `data/DAVIS` (Section 2).
2. Create the required conda envs and install dependencies (Section 5).
3. Edit `configs/compare.yaml` (task / pipelines / overwrite / out_root).
4. From repo root:

```bash
python -m object_removal.cli.compare
```

By default, `compare` reads `configs/compare.yaml`. If `out_root` is omitted, it defaults to:
`outputs/compare/<DAVIS sequence name>` (e.g. `davis:bmx-trees` → `outputs/compare/bmx-trees`).

---

## 1. Repository layout

```
src/object_removal/        # main package: cli / stages / methods / io
modules/                  # third-party: VGGT4D, SAM2, SAM3, ProPainter, DiffuEraser, Track-Anything…
configs/                  # compare.yaml + pipelines.yaml + env_map.json
ckpts/                    # model weights (not tracked)
data/DAVIS/               # DAVIS dataset (not tracked)
outputs/compare/          # default outputs (override via compare.yaml)
```

---

## 2. Data: DAVIS

Default `davis_root` is `data/DAVIS`. Expected structure:

```
data/DAVIS/
  JPEGImages/480p/<SEQ>/*.jpg
  Annotations_unsupervised/480p/<SEQ>/*.png   # or Annotations/480p/<SEQ>/*.png
```

Task syntax: `task: davis:<SEQ>` (e.g. `davis:bmx-trees`).

---

## 3. Weights: `ckpts/`

Place weights under these default paths (relative to repo root):

- **VGGT4D**: `ckpts/vggt4d/model_tracker_fixed_e20.pt`
  - If missing, `modules/VGGT4D/demo_vggt4d.py` may attempt to download it into `ckpts/vggt4d/` (network required), or you can place it manually.
- **SAM2**: put SAM2 checkpoint(s) under `ckpts/sam2/` (e.g. `sam2.1_hiera_large.pt`)
- **SAM3**: `ckpts/sam3/sam3.pt`
- **YOLOv8-seg**: `ckpts/yolo/yolov8n-seg.pt`
- **DiffuEraser bundle**: under `ckpts/diffueraser/weights/`:
  - `stable-diffusion-v1-5/`
  - `sd-vae-ft-mse/`
  - `diffuEraser/`
  - `propainter/`
  - `PCM_Weights/` (e.g. `PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors`)
- **Track-Anything**: follow its README and centralize checkpoints under `ckpts/trackanything/`.

---

## 4. Config-first workflow (avoid long CLI commands)

### 4.1 `configs/compare.yaml` — one “run job” config

This replaces long commands such as:

```bash
python -m object_removal.cli.compare --task davis:bmx-trees --pipelines ... --out_root ...
```

You typically edit:

- **`task`**
- **`pipelines`** (pipeline ids defined in `configs/pipelines.yaml`)
- **`overwrite`** (recommend `true` while iterating)
- **`out_root`** (optional; omit to use the default `outputs/compare/<SEQ>`)

### 4.2 `configs/pipelines.yaml` — pipelines + all tunable parameters

Each pipeline contains required fields `mask` / `track` / `inpaint`, plus optional blocks (all defined in YAML, not CLI):

- `vggt4d:` (includes `dyn_threshold_scale`, etc.)
- `sam3:`
- `diffueraser:` / `propainter:`

The YAML documents supported keys at the top.

### 4.3 `configs/env_map.json` — method → conda env

With default `env_policy: auto`, `compare` runs stages via:

`conda run -n <env> python -m object_removal.cli.<stage> ...`

Mapping lives in `configs/env_map.json`. Empty string `""` means: run that stage in the **current** Python environment running `compare`.

---

## 5. Per-conda-env installation (based on each module’s README)

### System dependencies

- **ffmpeg**: strongly recommended (e.g. `apt install ffmpeg`). The pipeline uses it to produce H.264 MP4 previews and to extract frames robustly.

### Common step (do this in every env)

```bash
cd /path/to/dynamic-object-removal
pip install -e .
pip install pyyaml
```

### 5.1 `vggt` env (VGGT4D)

Based on `modules/VGGT4D/README.md` and `modules/VGGT4D/requirements.txt`:

```bash
conda activate vggt
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/VGGT4D/requirements.txt
```

VGGT4D upstream suggests specific PyTorch/CUDA builds; install PyTorch matching your driver/CUDA.

### 5.2 `sam2` env (SAM2)

Based on `modules/sam2/README.md`:

```bash
conda activate sam2
cd /path/to/dynamic-object-removal
pip install -e .
pip install -e modules/sam2
```

Follow the upstream SAM2 README for the required PyTorch/CUDA toolkit (SAM2 may compile CUDA extensions).

### 5.3 `sam3` env (SAM3)

Based on `modules/sam3/README.md`:

```bash
conda activate sam3
cd /path/to/dynamic-object-removal
pip install -e .
pip install -e modules/sam3
```

Upstream SAM3 may require Hugging Face access/login to download checkpoints; you can also place a local checkpoint at `ckpts/sam3/sam3.pt`.

### 5.4 `propainter` env (ProPainter)

Based on `modules/ProPainter/README.md` and `modules/ProPainter/requirements.txt`:

```bash
conda activate propainter
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/ProPainter/requirements.txt
```

Weights may auto-download on first inference (or place them under `ckpts/propainter/weights/`).

### 5.5 `diffueraser` env (DiffuEraser)

Based on `modules/DiffuEraser/README.md` and `modules/DiffuEraser/requirements.txt`:

```bash
conda activate diffueraser
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/DiffuEraser/requirements.txt
```

Then prepare `ckpts/diffueraser/weights/` (Stable Diffusion, VAE, DiffuEraser, ProPainter, PCM_Weights).

### 5.6 `trackanything` env (Track-Anything)

Based on `modules/Track-Anything/README.md` and `modules/Track-Anything/requirements.txt`:

```bash
conda activate trackanything
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/Track-Anything/requirements.txt
```

---

## 6. Run

### One-command compare (recommended)

```bash
python -m object_removal.cli.compare
```

Outputs:

```
<out_root>/
  meta/                # compare_run.json / task.json / pipelines.json
  runs/<pipeline_id>/  # per-pipeline artifacts
  summary/             # combined.csv / combined.md (no experiment_name column)

**Reading `combined` metrics**

- **`mask_score`**: with per-frame pred vs GT masks, **`(mask_jm + mask_fm + mask_fr) / 3`** (`mask_fm` = boundary F-mean, `mask_fr` = boundary F-recall). If metrics come **only from a DAVIS summary CSV** (no FM/FR), **`(mask_jm + mask_jr) / 2`**.
- **`quality_score`**: currently **always `null`**—no bundled reference-free composite (`quality_score_source` = `disabled_no_single_reference_metric`). Raw terms (`bg_l1_mean`, temporal warp, Laplacian, optional BRISQUE) remain in `metrics_summary.json` for manual inspection or a future scorer.
- **Higher is better**: `mask_jm`, `mask_jr`, `mask_fm`, `mask_fr`, `mask_score`.
- **Lower is better**: `bg_l1_mean`, `temporal_warp_error_mean`, `temporal_warp_error_hole_mean`. `flow_consistency_mean`: **lower is smoother** on the background band (legacy naming).
```

### Manual per-stage (debug)

```bash
python -m object_removal.cli.mask    --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method vggt4d --repo_root .
python -m object_removal.cli.track   --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam3 --repo_root .
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method diffueraser
python -m object_removal.cli.eval    --run_dir runs/demo --pred_mask_dir runs/demo/track/masks_binary --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees --pred_frames_dir runs/demo/inpaint/frames --source_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```

---

## 7. Project conventions

Repo-specific rules: `AGENTS.md`. Design notes: `PLAN.md`.


# Dynamic Object Removal — Deployment & End-to-End Guide

This repository implements a modular video object removal pipeline split into four explicit stages:

**mask → track → inpaint → eval**

- **Core pipeline code**: `src/object_removal/`
- **Vendored third-party methods**: `modules/`
- **Run configuration**: `configs/compare.yaml` (one run) and `configs/pipelines.yaml` (pipelines + params)

[Chinese version](README_zh.md)

---

## 0. Quick start (recommended)

1. Prepare DAVIS under `data/DAVIS` (Section 2).
2. Prepare model checkpoints under `ckpts`, you can download from [Drive](https://drive.google.com/drive/folders/1iLGFL-ASrsymbmVhEBDXMnfaSD6uNd_M?usp=sharing)
3. Create the required conda envs and install dependencies (Section 5).
4. Edit `configs/compare.yaml` (task / pipelines / overwrite / out_root).
5. From repo root:

```bash
python -m object_removal.cli.compare
```

By default, `compare` reads `configs/compare.yaml`. If `out_root` is omitted, it defaults to:
`outputs/compare/<DAVIS sequence name>` (e.g. `davis:bmx-trees` → `outputs/compare/bmx-trees`).

---

## 1. Repository layout

```
src/object_removal/        # main package: cli / stages / methods / io
modules/                  # third-party: VGGT4D, SAM2, SAM3, ProPainter, DiffuEraser, xmem_tracker…
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
- **XMem**: place weights under `ckpts/xmem/` (default: `ckpts/xmem/XMem-s012.pth`; `sam_auto` init also uses `ckpts/xmem/sam_vit_*.pth`).

---

## 4. Config-first workflow (avoid long CLI commands)

### 4.1 `configs/compare.yaml` — one “run job” config

This replaces long commands such as:

```bash
python -m object_removal.cli.compare --task davis:bmx-trees --pipelines ... --out_root ...
```

You typically edit:

- **`task`**
- **`pipelines`** (pipeline ids defined in `configs/pipelines.yaml`; use this when you want a fixed subset)
- **`run_all_pipelines`** (mutually exclusive with `pipelines`; equivalent to `--all`)
- **`only_stage`** (optional `mask` / `track` / `inpaint` / `eval`; `eval` may iterate over multiple pipelines, the others require exactly one)
- **`overwrite`** (recommend `true` while iterating)
- **`out_root`** (optional; omit to use the default `outputs/compare/<SEQ>`)
- **`export_mask_vis`** plus optional **`mask_vis_fps`** / **`mask_vis_alpha`** (write a track overlay preview MP4)
- **`env_policy`** (`auto`, `force_multi`, or `force_single`, matching the compare CLI)

### 4.2 `configs/pipelines.yaml` — pipelines + all tunable parameters

Optional root key **`parameters`** holds shared option dicts per module (`vggt4d`, `sam3`, `xmem`, `diffueraser`, `propainter`). Each pipeline entry is merged shallowly: pipeline-specific blocks override the same keys from `parameters`. For apples-to-apples comparison, you can keep all tunables in `parameters` and reduce each pipeline to only `mask` / `track` / `inpaint`. The legacy key `defaults` is still read if present; `parameters` wins on conflicts.

Each pipeline contains required fields `mask` / `track` / `inpaint`. Optional per-pipeline override blocks are still supported, but they are not required when you keep everything in `parameters`:

- `vggt4d:` (includes `dyn_threshold_scale`, etc.)
- `sam3:`
- `diffueraser:` / `propainter:`

The YAML documents supported keys at the top and groups them by stage. In practice you usually keep pipeline ids stable and only tune the nested option blocks.

### 4.3 `configs/env_map.json` — method → conda env

With default `env_policy: auto`, `compare` runs stages via:

`conda run -n <env> python -m object_removal.cli.<stage> ...`

Mapping lives in `configs/env_map.json`. Empty string `""` means: run that stage in the **current** Python environment running `compare`.

Minimal example:

```json
{
  "mask": { "vggt4d": "vggt" },
  "track": { "sam3": "sam3", "identity": "" },
  "inpaint": { "diffueraser": "diffueraser" },
  "eval": { "default": "" }
}
```

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

### 5.6 `xmem` env (XMem)

The batch path now uses only the vendored XMem runtime files under `modules/xmem_tracker/`. Install the minimal runtime dependencies in that env:

```bash
conda activate xmem
cd /path/to/dynamic-object-removal
pip install -e .
pip install numpy pyyaml torch torchvision
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



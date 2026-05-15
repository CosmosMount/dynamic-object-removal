# DynaClear

Fully automatic video object removal and inpainting. Pipeline: **mask → track → inpaint → eval**.

Motion priors (VGGT4D) initialize dual-tracker propagation (XMem, SAM 3); reliability-aware vote fusion yields masks for DiffuEraser or ProPainter inpainting.

| Path | Role |
|------|------|
| `src/object_removal/` | CLI, stages, methods |
| `modules/` | VGGT4D, SAM3, ProPainter, DiffuEraser, xmem_tracker, FAST-VQA-and-FasterVQA, … |
| `configs/` | `compare.yaml`, `pipelines.yaml`, `env_map.json` |

[中文](README_zh.md)

---

## Setup

1. DAVIS under `data/DAVIS` (§2).
2. Checkpoints under `ckpts/` ([Drive](https://drive.google.com/drive/folders/1iLGFL-ASrsymbmVhEBDXMnfaSD6uNd_M?usp=sharing)).
3. Conda environments per §5.
4. `configs/compare.yaml`: `task`, `pipelines`, `overwrite`, optional `out_root`.

```bash
python -m object_removal.cli.compare
```

Default config: `configs/compare.yaml`. Default output: `outputs/compare/<SEQ>` when `out_root` is unset (`davis:bmx-trees` → `outputs/compare/bmx-trees`).

---

## Repository layout

```
src/object_removal/
modules/
configs/
ckpts/
data/DAVIS/
outputs/compare/
```

---

## Data (DAVIS)

Root: `data/DAVIS`.

```
data/DAVIS/
  JPEGImages/480p/<SEQ>/*.jpg
  Annotations_unsupervised/480p/<SEQ>/*.png
```

Task: `task: davis:<SEQ>` (e.g. `davis:bmx-trees`).

---

## Checkpoints (`ckpts/`)

| Component | Path |
|-----------|------|
| VGGT4D | `ckpts/vggt4d/model_tracker_fixed_e20.pt` (auto-download via `demo_vggt4d.py` if absent) |
| SAM2 | `ckpts/sam2/` |
| SAM3 | `ckpts/sam3/sam3.pt` |
| YOLOv8-seg | `ckpts/yolo/yolov8n-seg.pt` |
| XMem | `ckpts/xmem/XMem-s012.pth` |
| DiffuEraser | `ckpts/diffueraser/weights/` (`stable-diffusion-v1-5/`, `sd-vae-ft-mse/`, `diffuEraser/`, `propainter/`, `PCM_Weights/`) |
| FastVQA / FasterVQA | `ckpts/fastvqa/` (paths in `modules/FAST-VQA-and-FasterVQA/options/fast/*.yml`) |

---

## Configuration

### `configs/compare.yaml`

Run descriptor: `task`, `pipelines` or `run_all_pipelines`, `only_stage` (`mask` / `track` / `inpaint` / `eval`), `overwrite`, `out_root`, `export_mask_vis`, `env_policy` (`auto` | `force_multi` | `force_single`).

### `configs/pipelines.yaml`

Each pipeline: `mask`, `track`, `inpaint`. Optional root `parameters` (per-module defaults); pipeline entries override. Legacy `defaults` supported; `parameters` takes precedence.

### `configs/env_map.json`

`env_policy: auto` dispatches stages via `conda run -n <env> python -m object_removal.cli.<stage>`. Empty env name `""` uses the interpreter running `compare`.

| Stage | Method | Env | Spec |
|-------|--------|-----|------|
| mask | `vggt4d`, `vggt_framewise` | `vggt` | `modules/VGGT4D/env-vggt.txt` |
| mask | `yolo_first`, `baseline_yolo_motion` | `sam3` | `modules/sam3/env-sam3.txt` |
| track | `sam3` | `sam3` | `modules/sam3/env-sam3.txt` |
| track | `xmem` | `xmem` | `modules/xmem_tracker/env-xmem.txt` |
| track | `identity`, `optflow` | current | — |
| inpaint | `propainter` | `propainter` | `modules/ProPainter/env-propainter.txt` |
| inpaint | `diffueraser` | `diffueraser` | `modules/DiffuEraser/env-diffueraser.txt` |
| inpaint | `baseline_handcrafted` | current | — |
| eval | `default` | current | `configs/envs/base.txt` |
| eval | `fast_vqa` | `fastvqa` | `modules/FAST-VQA-and-FasterVQA/env-fastvqa.txt` |

`env_policy: force_single` runs all stages in one environment. With `eval.fast_vqa: true`, `compare` also requires the `fastvqa` env (VQA subprocess).

---

## Environments

`modules/*/env-*.txt`: pip dependencies; PyTorch installed per CUDA. System: `ffmpeg`.

**Base** (`compare`, eval):

```bash
conda create -n dynaclear-base python=3.10 -y
conda activate dynaclear-base
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r configs/envs/base.txt
```

**Backend** (example `vggt`):

```bash
conda create -n vggt python=3.10 -y
conda activate vggt
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r modules/VGGT4D/env-vggt.txt
```

| Env | Python | Spec |
|-----|--------|------|
| `dynaclear-base` | 3.10 | `configs/envs/base.txt` |
| `vggt` | 3.10 | `modules/VGGT4D/env-vggt.txt` |
| `sam3` | 3.12 | `modules/sam3/env-sam3.txt` |
| `xmem` | 3.10 | `modules/xmem_tracker/env-xmem.txt` |
| `propainter` | 3.10 | `modules/ProPainter/env-propainter.txt` |
| `diffueraser` | 3.10 | `modules/DiffuEraser/env-diffueraser.txt` |
| `fastvqa` | 3.10 | `modules/FAST-VQA-and-FasterVQA/env-fastvqa.txt` |

`compare` verifies required envs exist before execution.

---

## Execution

```bash
python -m object_removal.cli.compare
```

Output:

```
<out_root>/
  meta/
  runs/<pipeline_id>/
  summary/          # combined.csv, combined.md
```

**Metrics** (`combined.csv`): `mask_score = (mask_jm + mask_jr + mask_fm + mask_fr) / 4`. Higher: `mask_jm`, `mask_jr`, `mask_fm`, `mask_fr`, `mask_score`. Lower: `bg_l1_mean`, `temporal_warp_error_mean`, `temporal_warp_error_hole_mean`. `quality_score` is null (`quality_score_source`: `disabled_no_single_reference_metric`).

**Per-stage CLI:**

```bash
python -m object_removal.cli.mask    --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method vggt4d --repo_root .
python -m object_removal.cli.track   --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam3 --repo_root .
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method diffueraser
python -m object_removal.cli.eval    --run_dir runs/demo --pred_mask_dir runs/demo/track/masks_binary --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees --pred_frames_dir runs/demo/inpaint/frames --source_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```


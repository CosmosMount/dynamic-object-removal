# DynaClear

全自动视频物体移除与修复。流水线：**mask → track → inpaint → eval**。

VGGT4D 运动先验初始化双路跟踪（XMem、SAM 3）；可靠性 vote fusion 生成掩码，由 DiffuEraser 或 ProPainter 完成修复。

| 路径 | 说明 |
|------|------|
| `src/object_removal/` | CLI、阶段、方法实现 |
| `modules/` | VGGT4D、SAM3、ProPainter、DiffuEraser、xmem_tracker、FAST-VQA-and-FasterVQA 等 |
| `configs/` | `compare.yaml`、`pipelines.yaml`、`env_map.json` |

[English](README.md)

---

## 部署

1. DAVIS 置于 `data/DAVIS`（§2）。
2. 权重置于 `ckpts/`（[Drive](https://drive.google.com/drive/folders/15S5f7VWr7tWAEa4yf3ZHxLC_Phlj7Cxm?usp=sharing)）。
3. 按 §5 创建 conda 环境。
4. 配置 `configs/compare.yaml`：`task`、`pipelines`、`overwrite`、可选 `out_root`。

```bash
python -m object_removal.cli.compare
```

默认读取 `configs/compare.yaml`。未设置 `out_root` 时输出至 `outputs/compare/<SEQ>`（如 `davis:bmx-trees` → `outputs/compare/bmx-trees`）。

---

## 目录结构

```
src/object_removal/
modules/
configs/
ckpts/
data/DAVIS/
outputs/compare/
```

---

## 数据（DAVIS）

根目录：`data/DAVIS`。

```
data/DAVIS/
  JPEGImages/480p/<SEQ>/*.jpg
  Annotations_unsupervised/480p/<SEQ>/*.png
```

任务：`task: davis:<SEQ>`（例：`davis:bmx-trees`）。

---

## 权重（`ckpts/`）

| 模块 | 路径 |
|------|------|
| VGGT4D | `ckpts/vggt4d/model_tracker_fixed_e20.pt`（缺失时 `demo_vggt4d.py` 可自动下载） |
| SAM3 | `ckpts/sam3/sam3.pt` |
| YOLOv8-seg | `ckpts/yolo/yolov8n-seg.pt` |
| XMem | `ckpts/xmem/XMem-s012.pth` |
| DiffuEraser | `ckpts/diffueraser/weights/`（`stable-diffusion-v1-5/`、`sd-vae-ft-mse/`、`diffuEraser/`、`propainter/`、`PCM_Weights/`） |
| FastVQA / FasterVQA | `ckpts/fastvqa/`（路径见 `modules/FAST-VQA-and-FasterVQA/options/fast/*.yml`） |

---

## 配置

### `configs/compare.yaml`

运行项：`task`、`pipelines` 或 `run_all_pipelines`、`only_stage`（`mask` / `track` / `inpaint` / `eval`）、`overwrite`、`out_root`、`export_mask_vis`、`env_policy`（`auto` | `force_multi` | `force_single`）。

### `configs/pipelines.yaml`

每条 pipeline 含 `mask`、`track`、`inpaint`。可选根字段 `parameters` 为模块级默认；pipeline 内同名字段覆盖。兼容旧键 `defaults`，冲突时以 `parameters` 为准。

### `configs/env_map.json`

`env_policy: auto` 下各阶段经 `conda run -n <env> python -m object_removal.cli.<stage>` 执行。空环境名 `""` 表示与运行 `compare` 的解释器相同。

| 阶段 | 方法 | 环境 | 依赖文件 |
|------|------|------|----------|
| mask | `vggt4d`, `vggt_framewise` | `vggt` | `modules/VGGT4D/env-vggt.txt` |
| mask | `yolo_first`, `baseline_yolo_motion` | `sam3` | `modules/sam3/env-sam3.txt` |
| track | `sam3` | `sam3` | `modules/sam3/env-sam3.txt` |
| track | `xmem` | `xmem` | `modules/xmem_tracker/env-xmem.txt` |
| track | `identity`, `optflow` | 当前 | — |
| inpaint | `propainter` | `propainter` | `modules/ProPainter/env-propainter.txt` |
| inpaint | `diffueraser` | `diffueraser` | `modules/DiffuEraser/env-diffueraser.txt` |
| inpaint | `baseline_handcrafted` | 当前 | — |
| eval | `default` | 当前 | `configs/envs/base.txt` |
| eval | `fast_vqa` | `fastvqa` | `modules/FAST-VQA-and-FasterVQA/env-fastvqa.txt` |

`env_policy: force_single` 在单一环境中执行全部阶段。`eval.fast_vqa: true` 时另需 `fastvqa` 环境（VQA 子进程）。

---

## 环境

`modules/*/env-*.txt` 为 pip 依赖；PyTorch 按 CUDA 单独安装。系统依赖：`ffmpeg`。

**Base**（`compare`、eval）：

```bash
conda create -n dynaclear-base python=3.10 -y
conda activate dynaclear-base
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r configs/envs/base.txt
```

**后端**（以 `vggt` 为例）：

```bash
conda create -n vggt python=3.10 -y
conda activate vggt
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r modules/VGGT4D/env-vggt.txt
```

| 环境 | Python | 依赖文件 |
|------|--------|----------|
| `dynaclear-base` | 3.10 | `configs/envs/base.txt` |
| `vggt` | 3.10 | `modules/VGGT4D/env-vggt.txt` |
| `sam3` | 3.12 | `modules/sam3/env-sam3.txt` |
| `xmem` | 3.10 | `modules/xmem_tracker/env-xmem.txt` |
| `propainter` | 3.10 | `modules/ProPainter/env-propainter.txt` |
| `diffueraser` | 3.10 | `modules/DiffuEraser/env-diffueraser.txt` |
| `fastvqa` | 3.10 | `modules/FAST-VQA-and-FasterVQA/env-fastvqa.txt` |

执行前 `compare` 校验所选 pipeline 所需 conda 环境是否存在。

---

## 运行

```bash
python -m object_removal.cli.compare
```

输出：

```
<out_root>/
  meta/
  runs/<pipeline_id>/
  summary/          # combined.csv, combined.md
```

**指标**（`combined.csv`）：`mask_score = (mask_jm + mask_jr + mask_fm + mask_fr) / 4`。越高越好：`mask_jm`、`mask_jr`、`mask_fm`、`mask_fr`、`mask_score`。越低越好：`bg_l1_mean`、`temporal_warp_error_mean`、`temporal_warp_error_hole_mean`。`quality_score` 恒为 null（`quality_score_source`: `disabled_no_single_reference_metric`）。

**分阶段 CLI：**

```bash
python -m object_removal.cli.mask    --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method vggt4d --repo_root .
python -m object_removal.cli.track   --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam3 --repo_root .
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method diffueraser
python -m object_removal.cli.eval    --run_dir runs/demo --pred_mask_dir runs/demo/track/masks_binary --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees --pred_frames_dir runs/demo/inpaint/frames --source_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```


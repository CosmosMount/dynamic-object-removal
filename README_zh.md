# 动态物体移除（Video Object Removal）部署与运行指南

本仓库将「视频物体移除」拆分为四个稳定阶段：

**mask → track → inpaint → eval**

核心代码位于 `src/object_removal/`；第三方方法源码位于 `modules/`；配置位于 `configs/`。

---

## 0. 快速开始（建议先跑通）

1. 准备 DAVIS 数据到 `data/DAVIS`（见下文）。
2. 按需创建 conda 环境并安装依赖（见下文「每个环境安装什么」）。
3. 编辑 `configs/compare.yaml`（一次运行的 task / pipelines / overwrite / out_root 等）。
4. 在仓库根目录执行：

```bash
python -m object_removal.cli.compare
```

默认会读取 `configs/compare.yaml`。如果 `compare.yaml` 里不写 `out_root`，则默认输出到：
`outputs/compare/<DAVIS序列名>`（例如 `davis:bmx-trees` → `outputs/compare/bmx-trees`）。

---

## 1. 仓库结构

```
src/object_removal/        # 主包：cli / stages / methods / io
modules/                  # 第三方：VGGT4D、SAM2、SAM3、ProPainter、DiffuEraser、Track-Anything…
configs/                  # compare.yaml（一次运行） + pipelines.yaml（pipeline 定义） + env_map.json（方法→conda env）
ckpts/                    # 模型权重（不入 git）
data/DAVIS/               # DAVIS 数据（不入 git）
outputs/compare/          # 默认实验输出（可在 compare.yaml 改 out_root）
```

---

## 2. 数据准备（DAVIS）

默认 `davis_root: data/DAVIS`，目录应包含：

```
data/DAVIS/
  JPEGImages/480p/<SEQ>/*.jpg
  Annotations_unsupervised/480p/<SEQ>/*.png   # 或 Annotations/480p/<SEQ>/*.png
```

任务写法：`task: davis:<SEQ>`，例如 `davis:bmx-trees`。

---

## 3. 权重准备（`ckpts/`）

按本仓库默认约定放置（相对仓库根）：

- **VGGT4D**：`ckpts/vggt4d/model_tracker_fixed_e20.pt`
  - 若缺失，`modules/VGGT4D/demo_vggt4d.py` 会尝试从 Hugging Face 下载到 `ckpts/vggt4d/`（需要联网或手动放置）。
- **SAM2**：按你选择的 checkpoint 放到 `ckpts/sam2/`（如 `sam2.1_hiera_large.pt`）
- **SAM3**：`ckpts/sam3/sam3.pt`
- **YOLOv8-seg**：`ckpts/yolo/yolov8n-seg.pt`
- **DiffuEraser**：`ckpts/diffueraser/weights/` 下放置：
  - `stable-diffusion-v1-5/`
  - `sd-vae-ft-mse/`
  - `diffuEraser/`
  - `propainter/`
  - `PCM_Weights/`（例如 `PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors`）
- **Track-Anything**：权重按其 README 要求放置（并在 `ckpts/trackanything/` 下集中管理）

---

## 4. 配置文件（强烈推荐按配置跑，不再堆 CLI 参数）

### 4.1 `configs/compare.yaml`（一次运行的“作业配置”）

用于替代类似下面的长命令：

```bash
python -m object_removal.cli.compare --task davis:bmx-trees --pipelines ... --out_root ...
```

你主要会改：

- **`task`**：如 `davis:bmx-trees`
- **`pipelines`**：要跑哪些 pipeline id（这些 id 在 `configs/pipelines.yaml` 里定义）
- **`overwrite`**：是否覆盖（建议调参阶段用 true）
- **`out_root`**：可省略；省略则默认 `outputs/compare/<SEQ>`

### 4.2 `configs/pipelines.yaml`（pipeline 定义 + 需要传入的参数都写这里）

每个 pipeline 形如：

```yaml
pipelines:
  vggt_sam3_diffueraser:
    mask: vggt4d
    track: sam3
    inpaint: diffueraser
    vggt4d:
      dyn_threshold_scale: 0.7
      max_frames_for_vggt: 20
      # ...
    sam3:
      checkpoint: ckpts/sam3/sam3.pt
      two_stage_anchor_idx: auto
      # ...
    diffueraser:
      video_length: 10
      # ...
```

其中 `vggt4d:` / `sam3:` / `diffueraser:` / `propainter:` 的可写字段已经在该 YAML 顶部注释里列出，并会被 `compare` 解析并传入各 stage。

### 4.3 `configs/env_map.json`（方法 → conda env）

`compare` 在默认 `env_policy: auto` 下，会对每个 stage/method 自动使用：
`conda run -n <env> python -m object_removal.cli.<stage> ...`

映射表在 `configs/env_map.json`。值为空字符串 `""` 表示该 stage 在当前运行 `compare` 的环境中执行（你需要自己保证依赖）。

---

## 5. 每个 Conda 环境安装什么（按 modules README 组织）

> 目标：让 `compare` 能跨环境运行，同时每个 env 都能 import 本仓库包与对应 modules 依赖。

### 通用：每个 env 都先装本仓库

在每个 conda 环境内都执行一次（推荐 editable）：

```bash
cd /path/to/dynamic-object-removal
pip install -e .
pip install pyyaml
```

### 5.1 `vggt` 环境（VGGT4D）

依据 `modules/VGGT4D/README.md` / `requirements.txt`：

```bash
conda activate vggt
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/VGGT4D/requirements.txt
```

> VGGT4D README 中给了特定的 torch/cu118 版本示例；实际请按你的 CUDA/驱动匹配安装 PyTorch。

### 5.2 `sam2` 环境（SAM2）

依据 `modules/sam2/README.md`：需要 `pip install -e .` 安装 SAM2 本体：

```bash
conda activate sam2
cd /path/to/dynamic-object-removal
pip install -e .
pip install -e modules/sam2
```

并按 SAM2 README 安装对应的 PyTorch、CUDA toolkit（如需编译 CUDA extension）。

### 5.3 `sam3` 环境（SAM3）

依据 `modules/sam3/README.md`：需要 `pip install -e .` 安装 SAM3 本体：

```bash
conda activate sam3
cd /path/to/dynamic-object-removal
pip install -e .
pip install -e modules/sam3
```

SAM3 README 还要求（可能）Hugging Face 授权/登录以下载 checkpoint；你也可以手动把 checkpoint 放到 `ckpts/sam3/sam3.pt`。

### 5.4 `propainter` 环境（ProPainter）

依据 `modules/ProPainter/README.md` / `requirements.txt`：

```bash
conda activate propainter
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/ProPainter/requirements.txt
```

权重可运行时自动下载（或手动放到 `ckpts/propainter/weights/`）。

### 5.5 `diffueraser` 环境（DiffuEraser）

依据 `modules/DiffuEraser/README.md` / `requirements.txt`（该模块对 torch/diffusers 等版本较敏感）：

```bash
conda activate diffueraser
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/DiffuEraser/requirements.txt
```

并准备 `ckpts/diffueraser/weights/` 的模型目录（见上文）。

建议系统安装 `ffmpeg`，用于生成/重编码可播放的 MP4。

### 5.6 `trackanything` 环境（Track-Anything）

依据 `modules/Track-Anything/README.md` / `requirements.txt`：

```bash
conda activate trackanything
cd /path/to/dynamic-object-removal
pip install -e .
pip install -r modules/Track-Anything/requirements.txt
```

---

## 6. 运行与产物

### 6.1 一键 compare（推荐）

```bash
python -m object_removal.cli.compare
```

输出结构：

```
<out_root>/
  meta/                # compare_run.json / task.json / pipelines.json
  runs/<pipeline_id>/  # 每条 pipeline 的 mask/track/inpaint/eval 产物
  summary/             # combined.csv / combined.md
```

### 6.2 手动分阶段（排障）

```bash
python -m object_removal.cli.mask   --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method vggt4d --repo_root .
python -m object_removal.cli.track  --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam3 --repo_root .
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method diffueraser
python -m object_removal.cli.eval   --run_dir runs/demo --pred_mask_dir runs/demo/track/masks_binary --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees --pred_frames_dir runs/demo/inpaint/frames --gt_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```

---

## 7. 维护与规范

仓库内的修改原则与协作约定见 `AGENTS.md`；总体设计与实现笔记见 `PLAN.md`。


# 动态物体移除（Video Object Removal）部署与运行指南

本仓库将「视频物体移除」拆分为四个稳定阶段：

**mask → track → inpaint → eval**

核心代码位于 `src/object_removal/`；第三方方法源码位于 `modules/`；配置位于 `configs/`。

---

## 0. 快速开始（建议先跑通）

1. 准备 DAVIS 数据到 `data/DAVIS`（见下文）。
2. 准备模型权重到 `ckpts`，可以从[Drive](https://drive.google.com/drive/folders/1iLGFL-ASrsymbmVhEBDXMnfaSD6uNd_M?usp=sharing)下载
3. 按需创建 conda 环境并安装依赖（见下文「每个环境安装什么」）。
4. 编辑 `configs/compare.yaml`（一次运行的 task / pipelines / overwrite / out_root 等）。
5. 在仓库根目录执行：

```bash
python -m object_removal.cli.compare
```

默认会读取 `configs/compare.yaml`。如果 `compare.yaml` 里不写 `out_root`，则默认输出到：
`outputs/compare/<DAVIS序列名>`（例如 `davis:bmx-trees` → `outputs/compare/bmx-trees`）。

---

## 1. 仓库结构

```
src/object_removal/        # 主包：cli / stages / methods / io
modules/                  # 第三方：VGGT4D、SAM2、SAM3、ProPainter、DiffuEraser、xmem_tracker…
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
- **XMem**：权重放到 `ckpts/xmem/`（默认 `ckpts/xmem/XMem-s012.pth`；若用 `sam_auto` 初始化，也会读取 `ckpts/xmem/sam_vit_*.pth`）

---

## 4. 配置文件（强烈推荐按配置跑，不再堆 CLI 参数）

### 4.1 `configs/compare.yaml`（一次运行的“作业配置”）

用于替代类似下面的长命令：

```bash
python -m object_removal.cli.compare --task davis:bmx-trees --pipelines ... --out_root ...
```

你主要会改：

- **`task`**：如 `davis:bmx-trees`
- **`pipelines`**：要跑哪些 pipeline id（这些 id 在 `configs/pipelines.yaml` 里定义；适合只跑固定子集）
- **`run_all_pipelines`**：与 `pipelines` 二选一；等价于 `--all`
- **`only_stage`**：可选 `mask` / `track` / `inpaint` / `eval`；其中 `eval` 可遍历多个 pipeline，其余阶段必须只选一个 pipeline
- **`overwrite`**：是否覆盖（建议调参阶段用 true）
- **`out_root`**：可省略；省略则默认 `outputs/compare/<SEQ>`
- **`export_mask_vis`** 以及可选的 **`mask_vis_fps`** / **`mask_vis_alpha`**：导出 track 结果的叠加预览 MP4
- **`env_policy`**：`auto`、`force_multi`、`force_single`，与 compare CLI 含义一致

### 4.2 `configs/pipelines.yaml`（pipeline 定义 + 需要传入的参数都写这里）

根级可选字段 **`parameters`**：按模块统一写 `vggt4d` / `sam3` / `xmem` / `diffueraser` / `propainter` 五块公共参数；每个 pipeline 与同名字段浅合并（pipeline 里写的键覆盖 `parameters`）。如果你想做公平对比实验，可以把所有参数都收敛到 `parameters`，让每条 pipeline 只保留 `mask` / `track` / `inpaint` 三项。旧键名 **`defaults`** 仍兼容；若同时存在，同名键以 `parameters` 为准。

每个 pipeline 最简可以只写成：

```yaml
pipelines:
  vggt4d_sam3_diffueraser:
    mask: vggt4d
    track: sam3
    inpaint: diffueraser
```

其中 `vggt4d:` / `sam3:` / `diffueraser:` / `propainter:` 的可写字段已经在该 YAML 顶部注释里按 stage 分组列出，并会被 `compare` 解析后传入各 stage。

### 4.3 `configs/env_map.json`（方法 → conda env）

`compare` 在默认 `env_policy: auto` 下，会对每个 stage/method 自动使用：
`conda run -n <env> python -m object_removal.cli.<stage> ...`

映射表在 `configs/env_map.json`。值为空字符串 `""` 表示该 stage 在当前运行 `compare` 的环境中执行（你需要自己保证依赖）。

最小示例：

```json
{
  "mask": { "vggt4d": "vggt" },
  "track": { "sam3": "sam3", "identity": "" },
  "inpaint": { "diffueraser": "diffueraser" },
  "eval": { "default": "" }
}
```

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

### 5.6 `xmem` 环境（XMem）

当前批处理链路只使用 `modules/xmem_tracker/` 下这套 vendored XMem 运行文件；该环境安装最小运行依赖即可：

```bash
conda activate xmem
cd /path/to/dynamic-object-removal
pip install -e .
pip install numpy pyyaml torch torchvision
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
  summary/             # combined.csv / combined.md（无 experiment_name 列）

**combined 指标怎么读**

- **`mask_score`**：**`(mask_jm + mask_jr + mask_fm + mask_fr) / 4`**。
- **`quality_score`**：当前**固定为 `null`**，不再做无参考合成（`quality_score_source` 为 `disabled_no_single_reference_metric`）。`bg_l1_mean`、时序 warp、Laplacian、BRISQUE 等仍写入 `metrics_summary.json` 供人工或后续模型使用。
- **越高越好**：`mask_jm`、`mask_jr`、`mask_fm`、`mask_fr`、`mask_score`。
- **越低越好**：`bg_l1_mean`、`temporal_warp_error_mean`、`temporal_warp_error_hole_mean`。`flow_consistency_mean` 为相邻帧光流在背景上的变化幅度，**越低越稳**。
```

### 6.2 手动分阶段（排障）

```bash
python -m object_removal.cli.mask   --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --method vggt4d --repo_root .
python -m object_removal.cli.track  --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --in_masks_dir runs/demo/mask/init/masks --method sam3 --repo_root .
python -m object_removal.cli.inpaint --run_dir runs/demo --frames_dir data/DAVIS/JPEGImages/480p/bmx-trees --masks_dir runs/demo/track/masks_binary --method diffueraser
python -m object_removal.cli.eval   --run_dir runs/demo --pred_mask_dir runs/demo/track/masks_binary --gt_mask_dir data/DAVIS/Annotations_unsupervised/480p/bmx-trees --pred_frames_dir runs/demo/inpaint/frames --source_frames_dir data/DAVIS/JPEGImages/480p/bmx-trees
```

可选 **FastVQA / FasterVQA**（无参考 inpaint 视频质量）：建议将 [FAST-VQA-and-FasterVQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) 克隆到 `modules/FAST-VQA-and-FasterVQA`，并按其 README 安装依赖与配置权重；这样本项目会自动发现该目录。也可在 `configs/compare.yaml` 中设置 `eval.fast_vqa_root`，或对上述 `eval` 命令追加 `--fast-vqa --fast-vqa-root /path/to/repo`。`vqa.py` 默认用当前解释器运行；若出现 `rc=-11`（多为 SIGSEGV，常见于 CUDA/torch/decord 与当前 conda 环境不匹配），可设置环境变量 `FAST_VQA_PYTHON` 指向已装好 fastvqa 依赖的 Python，或将 `eval.fast_vqa_device` 设为 `cpu`（本仓库也会在 cuda 子进程异常退出时自动尝试一次 CPU）。分数写入每条 run 的 `eval/metrics_summary.json` 字段 `fast_vqa_score`（约 0–1，越大越好），`compare` 汇总表 `combined.csv` 中列为 `fast_vqa_score`。

---



# page/assets 媒体命名

页面使用相对路径 `./assets/...`。

## 示意图

| 文件 | 说明 |
|------|------|
| `pipeline.png` | 六阶段总览 |
| `2stage.png` | SAM3 两阶段锚定示意 |

## Inpainting 成片（最终去物体视频）

命名：`inpaint_<序列名>_<pipeline_id>.mp4`

| 管线 | 源文件（`outputs/compare/<seq>/runs/<pid>/...`） |
|------|--------------------------------------------------|
| `vggt4d_sam3_diffueraser`、`vggt4d_diffueraser` | `inpaint/diffueraser/diffueraser_result.mp4` |
| `vggt4d_trackanything`、`vggt4d_sam3_propainter`、`yolo_sam2` | `inpaint/propainter/<序列名>/inpaint_out.mp4` |

示例：

```bash
SEQ=bear
PID=vggt4d_sam3_diffueraser
cp "outputs/compare/$SEQ/runs/$PID/inpaint/diffueraser/diffueraser_result.mp4" \
   "page/assets/inpaint_${SEQ}_${PID}.mp4"
```

## mask_vis（掩码传播可视化）

仅 **tennis**、**scooter-board** 在页面上展示（命名不变）：

`mask_vis_<序列名>_<pipeline_id>.mp4` ← `runs/<pid>/track/mask_vis.mp4`

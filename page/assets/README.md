# page/assets 媒体命名

所有视频与示意图放在本目录；页面使用相对路径 `./assets/...`。

## 示意图（已纳入版本库）

| 文件 | 说明 |
|------|------|
| `pipeline.png` | 六阶段总览（VGGT4D → init → SAM3 → 后处理 → DiffuEraser） |
| `2stage.png` | SAM3 两阶段自动锚定示意 |

## mask_vis 视频

命名：`mask_vis_<序列名>_<pipeline_id>.mp4`

源文件（本机，默认在 `.gitignore` 的 `outputs/` 下）：

```text
outputs/compare/<序列名>/runs/<pipeline_id>/track/mask_vis.mp4
```

当前页面展示的管线 id：

- `vggt4d_sam3_diffueraser`（Ours）
- `vggt4d_trackanything`
- `vggt4d_diffueraser`
- `vggt4d_sam3_propainter`
- `yolo_sam2`

序列：`bmx-trees`、`tennis`、`scooter-board`。

更新时可在仓库根执行（示例）：

```bash
SEQ=tennis
PID=vggt4d_sam3_diffueraser
cp "outputs/compare/$SEQ/runs/$PID/track/mask_vis.mp4" \
   "page/assets/mask_vis_${SEQ}_${PID}.mp4"
```

## 可选：补全结果视频

若要在页面上增加 inpaint 成片，建议命名：

`inpaint_<序列名>_<pipeline_id>.mp4`

源路径随管线不同，例如：

`outputs/compare/<序列>/runs/<pipeline_id>/inpaint/diffueraser/diffueraser_result.mp4`

大文件请考虑 [Git LFS](https://git-lfs.github.com/) 或压缩码率后再提交。

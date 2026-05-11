# page/assets 媒体

- **图片**：`images/`（如 `pipeline.png`、`2stage.png`、`logo.png`、`logo-title.png`）
- **视频**：`video/`（`inpaint_<序列>_<管线>.mp4`、`mask_vis_<序列>_<管线>.mp4`）

## Inpainting 成片

命名：`inpaint_<序列名>_<pipeline_id>.mp4`

| 管线 | 源路径（`outputs/compare/<seq>/runs/<pid>/...`） |
|------|--------------------------------------------------|
| `vggt4d_sam3_diffueraser`、`vggt4d_diffueraser` | `inpaint/diffueraser/diffueraser_result.mp4` |
| `vggt4d_xmem`、`vggt4d_sam3_propainter`、`yolo_sam2` | `inpaint/propainter/<序列名>/inpaint_out.mp4` |

## mask_vis

`mask_vis_<序列>_<pipeline_id>.mp4` ← `runs/<pid>/track/mask_vis.mp4`

## Favicon

浏览器标签图标使用 `images/logo.png`。

# GitHub Pages 站点（`page/`）

本地预览：

```bash
python -m http.server 8000 --directory page
```

浏览器打开 `http://127.0.0.1:8000/`。

## 发布到 GitHub Pages

1. 推送本仓库后，在 GitHub 打开 **Settings → Pages**。
2. **Build and deployment → Source** 选择 **GitHub Actions**（不要选 “Deploy from a branch” 的 `/docs`，本站根目录为 `page/`）。
3. 使用工作流 [.github/workflows/pages.yml](../.github/workflows/pages.yml)：将 `page/` 作为静态产物上传。
4. 首次使用 Actions 部署时，按 GitHub 提示完成 **github-pages** environment 授权；部署完成后页面 URL 一般为 `https://<用户或组织>.github.io/dynamic-object-removal/`（以仓库 Settings 中显示为准）。

## 媒体与命名

见 [assets/README.md](assets/README.md)。

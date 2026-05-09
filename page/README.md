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

### 若曾出现 `configure-pages` / `Get Pages site failed` / `Not Found`

- **必须先**在 **Settings → Pages** 里把 **Source 选成「GitHub Actions」** 并保存，仓库里才会存在 Pages 站点记录；否则部分 Actions 去调 Pages API 会得到 404。
- 本仓库工作流已**去掉** `actions/configure-pages` 一步（纯静态 `page/` 不需要它），一般不再触发该错误。
- 若仍要用 `enablement: true` 自动开 Pages：官方说明该选项需要 **非 `GITHUB_TOKEN` 的 token**（如带 `repo` / Pages 写权限的 PAT），否则无法代你开启。

## 媒体与命名

见 [assets/README.md](assets/README.md)。

# Git 远程仓库管理方案

**日期**: 2026-04-09
**仓库**: F5-TTS

---

## 1. 背景与原因

F5-TTS 的原始仓库 `SWivid/F5-TTS` 是一个活跃的开源项目，有多个核心贡献者持续提交代码。我们的开发策略是：

1. 在本地基于 `main` 分支开发新功能
2. 新功能需要经过充分调试和验证
3. 稳定后再向原仓库提交 PR

**问题**: 直接推送到 `origin`（原仓库）没有权限，且不成熟代码会影响上游。

**解决**: Fork 原仓库到自己的 GitHub 账号，将 fork 作为日常开发的推送目标。

---

## 2. 实施步骤（已完成）

### 2.1 Fork 原仓库

在 GitHub 上 fork `SWivid/F5-TTS` → 生成 `hefengcn/F5-TTS`。

### 2.2 调整 Remote 配置

```bash
# 将原 origin 重命名为 upstream（保留与原仓库的关联）
git remote rename origin upstream

# 添加自己的 fork 为新的 origin
git remote add origin git@github.com:hefengcn/F5-TTS.git

# 设置 main 分支跟踪 origin
git push -u origin main
```

### 2.3 最终 Remote 配置

```
origin    git@github.com:hefengcn/F5-TTS.git  (fetch)  ← 你的 fork，日常推送
origin    git@github.com:hefengcn/F5-TTS.git  (push)
upstream  git@github.com:SWivid/F5-TTS.git    (fetch)  ← 原仓库，同步上游
upstream  git@github.com:SWivid/F5-TTS.git    (push)
```

---

## 3. 日常开发流程

### 3.1 开发新功能

```bash
# 基于最新 main 创建功能分支
git checkout -b feature/xxx

# 开发、提交...
git add <files>
git commit -m "描述"

# 推送到自己的 fork
git push -u origin feature/xxx
```

### 3.2 直接在 main 上开发（小改动）

```bash
# 提交并推送到自己的 fork
git add <files>
git commit -m "描述"
git push
```

### 3.3 同步上游更新

```bash
# 拉取原仓库最新代码
git fetch upstream

# 合并到本地 main
git checkout main
git merge upstream/main

# 推送到自己的 fork（保持 fork 同步）
git push
```

如果有冲突，手动解决后再提交。

### 3.4 向原仓库贡献代码（PR）

```bash
# 1. 确保功能分支已推送到 fork
git push origin feature/xxx

# 2. 在 GitHub 上创建 PR:
#    从 hefengcn/F5-TTS:feature/xxx → SWivid/F5-TTS:main
#    URL: https://github.com/SWivid/F5-TTS/compare/main...hefengcn:F5-TTS:feature/xxx
```

也可以在 fork 仓库页面点击 "Contribute" → "Open pull request"。

### 3.5 清理已合并的功能分支

```bash
# PR 被合并后，删除本地和远程分支
git branch -d feature/xxx
git push origin --delete feature/xxx
```

---

## 4. 注意事项

1. **不要 `git push upstream`** — 没有权限，也不应该直接推送
2. **定期同步 upstream** — 避免本地 main 与上游差距过大，减少合并冲突
3. **功能开发用分支** — 大功能在独立分支开发，保持 main 可随时同步上游
4. **.claude/ 和 .mcp.json 不提交** — 这些是 Claude Code 本地配置，已在 .gitignore 中或保持 untracked
5. **提交信息风格** — 参考原仓库风格：中文或英文均可，简洁描述变更内容

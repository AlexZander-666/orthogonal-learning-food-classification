@echo off
chcp 65001 >nul
echo ========================================
echo   GitHub 仓库上传脚本
echo ========================================
echo.
echo 正在准备上传到：
echo https://github.com/blackwhitez246/lightweight-food-classification
echo.
echo ========================================
echo.

REM 检查是否已初始化 Git
if not exist ".git" (
    echo [1/5] 初始化 Git 仓库...
    git init
    echo.
) else (
    echo [1/5] Git 仓库已存在，跳过初始化
    echo.
)

REM 检查是否已添加远程仓库
git remote -v | findstr "origin" >nul 2>&1
if errorlevel 1 (
    echo [2/5] 添加远程仓库...
    git remote add origin https://github.com/blackwhitez246/lightweight-food-classification.git
    echo.
) else (
    echo [2/5] 远程仓库已存在
    echo.
)

REM 添加文件
echo [3/5] 添加文件到 Git...
git add .
echo.

REM 提交
echo [4/5] 提交更改...
git commit -m "Initial commit: Lightweight food classification with knowledge distillation and attention mechanisms"
echo.

REM 推送
echo [5/5] 推送到 GitHub...
git branch -M main
git push -u origin main
echo.

if errorlevel 1 (
    echo ========================================
    echo   ❌ 上传失败！
    echo ========================================
    echo.
    echo 可能的原因：
    echo 1. 仓库不存在 - 需要先在 GitHub 创建仓库
    echo 2. 没有权限 - 需要登录 GitHub 账号
    echo 3. 网络问题 - 检查网络连接
    echo.
    echo 解决方案：
    echo 1. 访问 https://github.com/new
    echo 2. 创建名为 "lightweight-food-classification" 的仓库
    echo 3. 重新运行此脚本
    echo.
    pause
    exit /b 1
) else (
    echo ========================================
    echo   ✅ 上传成功！
    echo ========================================
    echo.
    echo 访问你的仓库：
    echo https://github.com/blackwhitez246/lightweight-food-classification
    echo.
    echo 下一步：
    echo 1. 检查仓库内容是否完整
    echo 2. 等待 arXiv 论文发布
    echo 3. 更新 README 中的 arXiv 链接
    echo.
    pause
)


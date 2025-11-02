@echo off
REM 编译论文脚本（Windows版）
REM 作者：Alex Zander

echo ==========================================
echo 编译 arXiv 论文
echo ==========================================

cd paper

echo.
echo [1/4] 第一次编译...
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ 完成
) else (
    echo   × 失败！查看错误：
    pdflatex main.tex
    pause
    exit /b 1
)

echo [2/4] 生成参考文献...
bibtex main >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ 完成
) else (
    echo   ⚠ 警告（可能正常）
)

echo [3/4] 第二次编译...
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ 完成
) else (
    echo   × 失败
    pause
    exit /b 1
)

echo [4/4] 第三次编译...
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ 完成
) else (
    echo   × 失败
    pause
    exit /b 1
)

echo.
echo ==========================================
echo √ 编译成功！
echo ==========================================
echo.
echo PDF文件: paper\main.pdf
echo.
echo 检查内容：
echo   - 作者：Alex Zander
echo   - 邮箱：21011149@mail.ecust.edu.cn
echo   - 单位：East China University of Science and Technology
echo.
echo 下一步：
echo   1. 打开 main.pdf 检查内容
echo   2. 手动打包文件提交到arXiv
echo.

pause


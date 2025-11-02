@echo off
REM ############################################################################
REM 论文改进实验一键运行脚本 (Windows版本)
REM 按照优先级顺序运行所有关键实验
REM ############################################################################

setlocal enabledelayedexpansion

REM 配置参数
set DATA_DIR=.\data
set CHECKPOINT_DIR=.\checkpoints
set RESULTS_DIR=.\experiments\results
set VIS_DIR=.\visualization\results

REM 实验配置
set NUM_STATISTICAL_RUNS=10
set EPOCHS_MAIN=30
set EPOCHS_CUB=30
set EPOCHS_HYPERPARAM=20
set SUBSET_RATIO=0.2

echo.
echo ================================
echo 论文改进实验自动化脚本
echo ================================
echo.

REM ############################################################################
REM 前置检查
REM ############################################################################

echo [检查] Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    exit /b 1
)
echo [成功] Python已安装

echo [检查] Python依赖包...
python -c "import torch; import torchvision; import numpy; import matplotlib; import scipy" >nul 2>&1
if errorlevel 1 (
    echo [错误] 缺少必要的Python包，请运行: pip install -r requirements.txt
    exit /b 1
)
echo [成功] 所有必要的Python包已安装

echo [检查] GPU可用性...
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > temp_device.txt
set /p DEVICE=<temp_device.txt
del temp_device.txt
if "%DEVICE%"=="cuda" (
    echo [成功] GPU可用
) else (
    echo [警告] GPU不可用，将使用CPU（速度会很慢）
)

REM 检查并创建目录
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%VIS_DIR%" mkdir "%VIS_DIR%"

REM ############################################################################
REM 检查模型checkpoints
REM ############################################################################

echo.
echo ================================
echo 检查模型Checkpoints
echo ================================
echo.

set BASELINE_CKPT=%CHECKPOINT_DIR%\baseline_best.pth
set TEACHER_CKPT=%CHECKPOINT_DIR%\teacher_best.pth
set SIMAM_CKPT=%CHECKPOINT_DIR%\simam_best.pth
set SIMAM_KD_CKPT=%CHECKPOINT_DIR%\simam_kd_best.pth

set NEED_TRAIN=0

if not exist "%BASELINE_CKPT%" (
    echo [警告] 基线模型不存在: %BASELINE_CKPT%
    set NEED_TRAIN=1
)

if not exist "%TEACHER_CKPT%" (
    echo [警告] 教师模型不存在: %TEACHER_CKPT%
    set NEED_TRAIN=1
)

if not exist "%SIMAM_CKPT%" (
    echo [警告] SimAM模型不存在: %SIMAM_CKPT%
    set NEED_TRAIN=1
)

if not exist "%SIMAM_KD_CKPT%" (
    echo [警告] SimAM+KD模型不存在: %SIMAM_KD_CKPT%
    set NEED_TRAIN=1
)

if !NEED_TRAIN!==1 (
    echo.
    echo [警告] 需要先训练基础模型，这将需要较长时间...
    set /p TRAIN_CONFIRM="是否现在训练？(y/n): "
    if /i "!TRAIN_CONFIRM!"=="y" (
        echo.
        echo ================================
        echo 训练基础模型
        echo ================================
        echo.
        
        REM 训练基线
        if not exist "%BASELINE_CKPT%" (
            echo [训练] 基线MobileNetV3...
            python train1.py --data-dir "%DATA_DIR%" --output-dir "%CHECKPOINT_DIR%" --epochs %EPOCHS_MAIN% --device %DEVICE%
        )
        
        REM 训练教师
        if not exist "%TEACHER_CKPT%" (
            echo [训练] ResNet-50教师...
            python train_teacher.py --data-dir "%DATA_DIR%" --output-dir "%CHECKPOINT_DIR%" --epochs %EPOCHS_MAIN% --device %DEVICE%
        )
        
        REM 训练SimAM
        if not exist "%SIMAM_CKPT%" (
            echo [训练] SimAM模型...
            python train_distillation.py --data-dir "%DATA_DIR%" --attention-type simam --output-dir "%CHECKPOINT_DIR%" --epochs %EPOCHS_MAIN% --device %DEVICE% --no-distillation
        )
        
        REM 训练SimAM+KD
        if not exist "%SIMAM_KD_CKPT%" (
            echo [训练] SimAM+KD模型...
            python train_distillation.py --data-dir "%DATA_DIR%" --attention-type simam --teacher-checkpoint "%TEACHER_CKPT%" --output-dir "%CHECKPOINT_DIR%" --epochs %EPOCHS_MAIN% --device %DEVICE%
        )
        
        echo [成功] 所有基础模型训练完成！
    ) else (
        echo [错误] 需要先训练基础模型才能继续。
        exit /b 1
    )
)

REM ############################################################################
REM 实验菜单
REM ############################################################################

echo.
echo ================================
echo 实验菜单
echo ================================
echo.
echo 请选择要运行的实验：
echo   1. [P0] Grad-CAM可视化分析 (~30分钟)
echo   2. [P0] CUB-200-2011泛化验证 (~8-10小时)
echo   3. [P1] 统计显著性检验 (~2-3天)
echo   4. [P2] 超参数交互分析 (~10-15小时)
echo   5. 运行所有实验（自动化）
echo   0. 退出
echo.

set /p CHOICE="请输入选项 (0-5): "

if "%CHOICE%"=="1" goto EXP1
if "%CHOICE%"=="2" goto EXP2
if "%CHOICE%"=="3" goto EXP3
if "%CHOICE%"=="4" goto EXP4
if "%CHOICE%"=="5" goto EXP_ALL
if "%CHOICE%"=="0" goto EXIT
echo [错误] 无效选项
goto EXIT

REM ############################################################################
REM 实验1: Grad-CAM可视化
REM ############################################################################
:EXP1
echo.
echo ================================
echo 实验1: Grad-CAM可视化分析
echo ================================
echo.

python visualization\gradcam_analysis.py ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%VIS_DIR%\gradcam" ^
    --baseline-checkpoint "%BASELINE_CKPT%" ^
    --simam-checkpoint "%SIMAM_CKPT%" ^
    --simam-kd-checkpoint "%SIMAM_KD_CKPT%" ^
    --num-samples 20

echo [成功] Grad-CAM可视化完成！
echo [提示] 请查看: %VIS_DIR%\gradcam\
goto SUMMARY

REM ############################################################################
REM 实验2: CUB-200泛化验证
REM ############################################################################
:EXP2
echo.
echo ================================
echo 实验2: CUB-200-2011泛化验证
echo ================================
echo.

if not exist "%DATA_DIR%\CUB_200_2011" (
    echo [错误] CUB-200-2011数据集不存在
    echo 请从以下地址下载:
    echo https://data.caltech.edu/records/65de6-vp158
    echo 然后解压到: %DATA_DIR%\CUB_200_2011\
    goto EXIT
)

python experiments\train_cub200.py ^
    --data-dir "%DATA_DIR%\CUB_200_2011" ^
    --output-dir "%RESULTS_DIR%\cub200" ^
    --device %DEVICE%

echo [成功] CUB-200实验完成！
echo [提示] 请查看: %RESULTS_DIR%\cub200\
goto SUMMARY

REM ############################################################################
REM 实验3: 统计显著性检验
REM ############################################################################
:EXP3
echo.
echo ================================
echo 实验3: 统计显著性检验
echo ================================
echo.

echo [警告] 这将进行 %NUM_STATISTICAL_RUNS% 次独立运行，需要很长时间...
set /p CONFIRM="确认继续？(y/n): "
if /i not "%CONFIRM%"=="y" (
    echo [取消] 已取消
    goto EXIT
)

python experiments\statistical_significance.py ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%RESULTS_DIR%\statistical" ^
    --num-runs %NUM_STATISTICAL_RUNS% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --device %DEVICE%

echo [成功] 统计显著性检验完成！
echo [提示] 请查看: %RESULTS_DIR%\statistical\
goto SUMMARY

REM ############################################################################
REM 实验4: 超参数交互分析
REM ############################################################################
:EXP4
echo.
echo ================================
echo 实验4: 超参数交互分析
echo ================================
echo.

python experiments\hyperparameter_interaction.py ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%RESULTS_DIR%\hyperparameter" ^
    --attention-type simam ^
    --epochs %EPOCHS_HYPERPARAM% ^
    --subset-ratio %SUBSET_RATIO% ^
    --device %DEVICE%

echo [成功] 超参数交互分析完成！
echo [提示] 请查看: %RESULTS_DIR%\hyperparameter\
goto SUMMARY

REM ############################################################################
REM 运行所有实验
REM ############################################################################
:EXP_ALL
echo.
echo ================================
echo 运行所有实验
echo ================================
echo.

echo [警告] 这将依次运行所有4个实验，总共需要约3-5天时间
set /p CONFIRM="确认继续？(y/n): "
if /i not "%CONFIRM%"=="y" (
    echo [取消] 已取消
    goto EXIT
)

REM 实验1
echo.
echo [1/4] Grad-CAM可视化...
python visualization\gradcam_analysis.py ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%VIS_DIR%\gradcam" ^
    --baseline-checkpoint "%BASELINE_CKPT%" ^
    --simam-checkpoint "%SIMAM_CKPT%" ^
    --simam-kd-checkpoint "%SIMAM_KD_CKPT%" ^
    --num-samples 20

REM 实验2
if exist "%DATA_DIR%\CUB_200_2011" (
    echo.
    echo [2/4] CUB-200泛化验证...
    python experiments\train_cub200.py ^
        --data-dir "%DATA_DIR%\CUB_200_2011" ^
        --output-dir "%RESULTS_DIR%\cub200" ^
        --device %DEVICE%
) else (
    echo [跳过] CUB-200实验（数据集未下载）
)

REM 实验3
echo.
echo [3/4] 统计显著性检验...
python experiments\statistical_significance.py ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%RESULTS_DIR%\statistical" ^
    --num-runs %NUM_STATISTICAL_RUNS% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --device %DEVICE%

REM 实验4
echo.
echo [4/4] 超参数交互分析...
python experiments\hyperparameter_interaction.py ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%RESULTS_DIR%\hyperparameter" ^
    --attention-type simam ^
    --epochs %EPOCHS_HYPERPARAM% ^
    --subset-ratio %SUBSET_RATIO% ^
    --device %DEVICE%

echo [成功] 所有实验完成！

REM ############################################################################
REM 完成总结
REM ############################################################################
:SUMMARY
echo.
echo ================================
echo 实验完成总结
echo ================================
echo.
echo 实验结果位置:
echo   - Grad-CAM可视化: %VIS_DIR%\gradcam\
echo   - CUB-200结果: %RESULTS_DIR%\cub200\
echo   - 统计显著性: %RESULTS_DIR%\statistical\
echo   - 超参数分析: %RESULTS_DIR%\hyperparameter\
echo.
echo 下一步:
echo   1. 查看生成的图表和数据
echo   2. 更新论文中的占位符 (XX.XX)
echo   3. 参考 EXPERIMENTS_GUIDE.md 了解如何使用结果
echo.
echo [成功] 所有任务完成！祝论文发表顺利！
goto END

:EXIT
echo [退出] 程序已退出

:END
endlocal
pause













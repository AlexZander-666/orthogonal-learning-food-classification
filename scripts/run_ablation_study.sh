#!/bin/bash
# 消融实验脚本
# 测试不同注意力机制和训练策略的影响

echo "=========================================="
echo "开始消融实验"
echo "=========================================="

# 设置参数
DATA_DIR="./data"
BATCH_SIZE=64
EPOCHS=30
OUTPUT_DIR="./ablation_results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 实验1: Baseline (MobileNetV3 without attention, without distillation)
echo ""
echo "实验1: Baseline - MobileNetV3 (无注意力，无蒸馏)"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type none \
    --output-dir $OUTPUT_DIR/baseline \
    --pretrained

# 实验2: MobileNetV3 + ECA (without distillation)
echo ""
echo "实验2: MobileNetV3 + ECA (无蒸馏)"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type eca \
    --output-dir $OUTPUT_DIR/eca_no_distill \
    --pretrained

# 实验3: MobileNetV3 + SimAM (without distillation)
echo ""
echo "实验3: MobileNetV3 + SimAM (无蒸馏)"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type simam \
    --output-dir $OUTPUT_DIR/simam_no_distill \
    --pretrained

# 实验4: MobileNetV3 + ECA + Distillation (完整方法)
echo ""
echo "实验4: MobileNetV3 + ECA + Distillation (完整方法)"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type eca \
    --teacher-checkpoint ./teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir $OUTPUT_DIR/eca_distill \
    --pretrained

# 实验5: MobileNetV3 + SimAM + Distillation (完整方法)
echo ""
echo "实验5: MobileNetV3 + SimAM + Distillation (完整方法)"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type simam \
    --teacher-checkpoint ./teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir $OUTPUT_DIR/simam_distill \
    --pretrained

# 实验6: MobileNetV3 + CBAM + Distillation
echo ""
echo "实验6: MobileNetV3 + CBAM + Distillation"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type cbam \
    --teacher-checkpoint ./teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir $OUTPUT_DIR/cbam_distill \
    --pretrained

# 实验7: MobileNetV3 + SE + Distillation
echo ""
echo "实验7: MobileNetV3 + SE + Distillation"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type se \
    --teacher-checkpoint ./teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir $OUTPUT_DIR/se_distill \
    --pretrained

# 实验8: MobileNetV3 + CoordAttention + Distillation
echo ""
echo "实验8: MobileNetV3 + CoordAttention + Distillation"
python train_distillation.py \
    --data-dir $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --attention-type coord \
    --teacher-checkpoint ./teacher_resnet50.pth \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir $OUTPUT_DIR/coord_distill \
    --pretrained

echo ""
echo "=========================================="
echo "消融实验完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

# 生成对比报告
python utils/generate_ablation_report.py --results-dir $OUTPUT_DIR



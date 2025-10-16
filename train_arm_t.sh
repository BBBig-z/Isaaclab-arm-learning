#!/bin/bash
# ARM-T 训练脚本
# 使用方法: ./train_arm_t.sh [选项]

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "isaaclab_env" ]; then
    echo "正在激活 isaaclab_env 环境..."
    eval "$(conda shell.bash hook)"
    conda activate isaaclab_env
    if [ $? -ne 0 ]; then
        echo "错误: 无法激活 isaaclab_env 环境"
        echo "请先运行: conda activate isaaclab_env"
        exit 1
    fi
fi

# 设置 Python 路径（确保 arm_t 模块可以被找到）
export PYTHONPATH="${PROJECT_ROOT}/source/ARM:${PYTHONPATH}"

echo "=========================================="
echo "ARM-T 六自由度机械臂训练脚本"
echo "=========================================="
echo ""

# 显示帮助信息
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "使用方法:"
    echo "  ./train_arm_t.sh [任务] [选项]"
    echo ""
    echo "可用任务 (Reach任务):"
    echo "  joint       - Reach任务，关节位置控制 (默认)"
    echo "  ik          - Reach任务，逆运动学控制"
    echo ""
    echo "可用任务 (Lift任务):"
    echo "  lift-joint  - Lift任务，关节位置控制"
    echo "  lift-ik     - Lift任务，逆运动学控制"
    echo ""
    echo "常用选项:"
    echo "  --headless              - 无图形界面模式"
    echo "  --num_envs N            - 并行环境数量 (默认: 2048)"
    echo "  --max_iterations N      - 最大训练迭代次数"
    echo "  --video                 - 录制训练视频"
    echo ""
    echo "恢复训练选项:"
    echo "  --resume                - 从最新checkpoint恢复训练"
    echo "  --load_run RUN_NAME     - 指定要恢复的运行名称"
    echo "  --checkpoint PATH       - 指定checkpoint文件路径"
    echo ""
    echo "示例:"
    echo "  ./train_arm_t.sh                                    # Reach任务，关节控制"
    echo "  ./train_arm_t.sh joint --headless                   # Reach任务，Headless模式"
    echo "  ./train_arm_t.sh ik --num_envs 2048                 # Reach任务，IK控制"
    echo "  ./train_arm_t.sh lift-joint --headless              # Lift任务，关节控制"
    echo "  ./train_arm_t.sh lift-ik --num_envs 2048            # Lift任务，IK控制"
    echo ""
    echo "恢复训练示例:"
    echo "  # 方法1: 自动恢复最新checkpoint"
    echo "  ./train_arm_t.sh joint --resume"
    echo ""
    echo "  # 方法2: 指定运行名称（自动找最新checkpoint）"
    echo "  ./train_arm_t.sh lift-ik --resume --load_run 2025-10-14_12-00-00"
    echo ""
    echo "  # 方法3: 指定具体checkpoint文件"
    echo "  ./train_arm_t.sh joint --resume --checkpoint logs/rsl_rl/arm_t_reach/2025-10-12_04-21-41/model_1000.pt"
    echo ""
    exit 0
fi

# 解析任务类型
TASK_TYPE="${1:-joint}"
shift || true

case "$TASK_TYPE" in
    joint)
        TASK="ARM-T-Reach-v0"
        TASK_NAME="reach"
        echo "✓ 任务: Reach - 关节位置控制"
        ;;
    ik)
        TASK="ARM-T-Reach-IK-v0"
        TASK_NAME="reach_ik"
        echo "✓ 任务: Reach - 逆运动学控制"
        ;;
    lift-joint|lift)
        TASK="ARM-T-Lift-Cube-v0"
        TASK_NAME="lift"
        echo "✓ 任务: Lift - 关节位置控制"
        ;;
    lift-ik)
        TASK="ARM-T-Lift-Cube-IK-v0"
        TASK_NAME="lift_ik"
        echo "✓ 任务: Lift - 逆运动学控制"
        ;;
    *)
        echo "错误: 未知任务类型 '$TASK_TYPE'"
        echo "运行 './train_arm_t.sh --help' 查看帮助"
        exit 1
        ;;
esac

echo "✓ 环境: $TASK"

# 检查是否启用了恢复训练
RESUME_FLAG=false
HAS_CHECKPOINT_ARG=false
for arg in "$@"; do
    if [ "$arg" == "--resume" ]; then
        RESUME_FLAG=true
    elif [ "$arg" == "--checkpoint" ] || [ "$arg" == "--load_run" ]; then
        HAS_CHECKPOINT_ARG=true
    fi
done

# 如果启用了 --resume 但没有指定具体checkpoint，显示交互式选择菜单
if [ "$RESUME_FLAG" == true ] && [ "$HAS_CHECKPOINT_ARG" == false ]; then
    echo ""
    echo "=========================================="
    echo "📦 选择要恢复的Checkpoint"
    echo "=========================================="
    echo ""
    
    # 确定日志目录
    case "$TASK_NAME" in
        reach)
            LOG_DIR="logs/rsl_rl/arm_t_reach"
            ;;
        reach_ik)
            LOG_DIR="logs/rsl_rl/arm_t_reach_ik"
            ;;
        lift)
            LOG_DIR="logs/rsl_rl/arm_t_lift"
            ;;
        lift_ik)
            LOG_DIR="logs/rsl_rl/arm_t_lift_ik"
            ;;
    esac
    
    # 查找所有checkpoint文件
    if [ ! -d "$LOG_DIR" ]; then
        echo "⚠ 未找到日志目录: $LOG_DIR"
        echo "请先运行训练生成checkpoint"
        exit 1
    fi
    
    # 查找所有model_*.pt文件并排序
    CHECKPOINTS=($(find "$LOG_DIR" -name "model_*.pt" | sort -V))
    
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "⚠ 未找到任何checkpoint文件"
        echo "请先运行训练生成checkpoint"
        exit 1
    fi
    
    echo "找到 ${#CHECKPOINTS[@]} 个可用的checkpoint:"
    echo ""
    
    # 显示checkpoint列表
    for i in "${!CHECKPOINTS[@]}"; do
        CHECKPOINT="${CHECKPOINTS[$i]}"
        RUN_DIR=$(dirname "$CHECKPOINT")
        RUN_NAME=$(basename "$RUN_DIR")
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")
        CHECKPOINT_SIZE=$(du -h "$CHECKPOINT" | cut -f1)
        
        # 提取迭代次数
        ITER=$(echo "$CHECKPOINT_NAME" | sed 's/model_\([0-9]*\).pt/\1/')
        
        printf "  [%2d] %s\n" "$((i+1))" "$CHECKPOINT_NAME"
        printf "      运行: %s\n" "$RUN_NAME"
        printf "      迭代: %s  |  大小: %s\n" "$ITER" "$CHECKPOINT_SIZE"
        echo ""
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 提示用户选择
    while true; do
        read -p "请选择checkpoint编号 [1-${#CHECKPOINTS[@]}] 或 'q' 退出: " choice
        
        if [ "$choice" == "q" ] || [ "$choice" == "Q" ]; then
            echo "已取消"
            exit 0
        fi
        
        # 检查输入是否为数字
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#CHECKPOINTS[@]} ]; then
            SELECTED_CHECKPOINT="${CHECKPOINTS[$((choice-1))]}"
            SELECTED_RUN=$(basename "$(dirname "$SELECTED_CHECKPOINT")")
            SELECTED_MODEL=$(basename "$SELECTED_CHECKPOINT")
            echo ""
            echo "✓ 已选择: $SELECTED_MODEL"
            echo "  运行: $SELECTED_RUN"
            echo "  路径: $SELECTED_CHECKPOINT"
            echo ""
            break
        else
            echo "⚠ 无效输入，请输入 1-${#CHECKPOINTS[@]} 之间的数字"
        fi
    done
    
    # 添加checkpoint参数到命令行（RSL-RL需要run名称和checkpoint文件名）
    set -- "$@" --load_run "$SELECTED_RUN" --checkpoint "$SELECTED_MODEL"
    echo "✓ 模式: 恢复训练（从checkpoint继续）"
elif [ "$RESUME_FLAG" == true ]; then
    echo "✓ 模式: 恢复训练（从checkpoint继续）"
else
    echo "✓ 模式: 从头开始训练"
fi
echo ""

# 记录训练开始时间（用于后续识别本次训练产生的文件）
TRAINING_START_TIME=$(date +%s)
TRAINING_DATE=$(date +%Y-%m-%d)

# 运行训练
echo "开始训练..."
echo "=========================================="
python3 scripts/rsl_rl/train.py --task "$TASK" "$@"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成"
else
    echo "✗ 训练出错（退出码: $TRAIN_EXIT_CODE）"
fi

echo ""

# 查找本次训练产生的文件
echo "=========================================="
echo "清理训练文件"
echo "=========================================="

# 确定日志子目录
case "$TASK_NAME" in
    reach)
        LOG_SUBDIR="arm_t_reach"
        ;;
    reach_ik)
        LOG_SUBDIR="arm_t_reach_ik"
        ;;
    lift)
        LOG_SUBDIR="arm_t_lift"
        ;;
    lift_ik)
        LOG_SUBDIR="arm_t_lift_ik"
        ;;
esac

# 查找在训练开始后创建的日志目录
TRAIN_LOGS_DIRS=()
if [ -d "logs/rsl_rl/${LOG_SUBDIR}" ]; then
    while IFS= read -r -d '' dir; do
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
        if [ "$DIR_MTIME" -ge "$TRAINING_START_TIME" ]; then
            TRAIN_LOGS_DIRS+=("$dir")
        fi
    done < <(find "logs/rsl_rl/${LOG_SUBDIR}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
fi

# 查找在训练开始后创建的输出目录
TRAIN_OUTPUT_DIRS=()
if [ -d "outputs/${TRAINING_DATE}" ]; then
    while IFS= read -r -d '' dir; do
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
        if [ "$DIR_MTIME" -ge "$TRAINING_START_TIME" ]; then
            TRAIN_OUTPUT_DIRS+=("$dir")
        fi
    done < <(find "outputs/${TRAINING_DATE}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
fi

# 显示找到的文件并询问是否保留
if [ ${#TRAIN_LOGS_DIRS[@]} -gt 0 ] || [ ${#TRAIN_OUTPUT_DIRS[@]} -gt 0 ]; then
    echo "本次训练产生的文件："
    echo ""
    
    for dir in "${TRAIN_LOGS_DIRS[@]}"; do
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  📁 $dir ($SIZE)"
    done
    
    for dir in "${TRAIN_OUTPUT_DIRS[@]}"; do
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  📁 $dir ($SIZE)"
    done
    
    echo ""
    echo -e "\033[1;33m是否保留本次训练内容？\033[0m"
    echo "  输入 'y' 保留，其他任意键删除"
    read -p "请选择 [y/N]: " -n 1 -r KEEP_TRAINING
    echo ""
    
    if [[ ! $KEEP_TRAINING =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "\033[1;33m正在删除本次训练文件...\033[0m"
        
        for dir in "${TRAIN_LOGS_DIRS[@]}"; do
            echo "  删除: $dir"
            trash-put "$dir"
        done
        
        for dir in "${TRAIN_OUTPUT_DIRS[@]}"; do
            echo "  删除: $dir"
            trash-put "$dir"
        done
        
        echo -e "\033[0;32m✓ 训练文件已删除\033[0m"
        echo ""
        exit 0
    else
        echo -e "\033[0;32m✓ 训练文件已保留\033[0m"
        echo ""
        echo "=========================================="
        echo ""
    fi
else
    echo "未找到本次训练产生的文件"
    echo ""
fi

# 根据任务类型显示不同的提示信息（仅在保留文件时显示）
case "$TASK_NAME" in
    reach)
        echo "日志位置: logs/rsl_rl/arm_t_reach/"
        echo ""
        echo "查看训练结果:"
        echo "  tensorboard --logdir logs/rsl_rl/arm_t_reach"
        echo ""
        echo "测试模型:"
        echo "  python3 scripts/rsl_rl/play.py --task ARM-T-Reach-Play-v0 \\"
        echo "      --checkpoint logs/rsl_rl/arm_t_reach/*/model_*.pt"
        echo ""
        echo "恢复训练:"
        echo "  ./train_arm_t.sh joint --resume"
        ;;
    reach_ik)
        echo "日志位置: logs/rsl_rl/arm_t_reach_ik/"
        echo ""
        echo "查看训练结果:"
        echo "  tensorboard --logdir logs/rsl_rl/arm_t_reach_ik"
        echo ""
        echo "测试模型:"
        echo "  python3 scripts/rsl_rl/play.py --task ARM-T-Reach-IK-Play-v0 \\"
        echo "      --checkpoint logs/rsl_rl/arm_t_reach_ik/*/model_*.pt"
        echo ""
        echo "恢复训练:"
        echo "  ./train_arm_t.sh ik --resume"
        ;;
    lift)
        echo "日志位置: logs/rsl_rl/arm_t_lift/"
        echo ""
        echo "查看训练结果:"
        echo "  tensorboard --logdir logs/rsl_rl/arm_t_lift"
        echo ""
        echo "测试模型:"
        echo "  python3 scripts/rsl_rl/play.py --task ARM-T-Lift-Cube-Play-v0 \\"
        echo "      --checkpoint logs/rsl_rl/arm_t_lift/*/model_*.pt"
        echo ""
        echo "恢复训练:"
        echo "  ./train_arm_t.sh lift-joint --resume"
        ;;
    lift_ik)
        echo "日志位置: logs/rsl_rl/arm_t_lift_ik/"
        echo ""
        echo "查看训练结果:"
        echo "  tensorboard --logdir logs/rsl_rl/arm_t_lift_ik"
        echo ""
        echo "测试模型:"
        echo "  python3 scripts/rsl_rl/play.py --task ARM-T-Lift-Cube-IK-Play-v0 \\"
        echo "      --checkpoint logs/rsl_rl/arm_t_lift_ik/*/model_*.pt"
        echo ""
        echo "恢复训练:"
        echo "  ./train_arm_t.sh lift-ik --resume"
        ;;
esac

echo ""


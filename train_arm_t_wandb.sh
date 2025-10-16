#!/bin/bash

# ARM-T WandB训练启动脚本
# 为所有任务提供便捷的WandB训练入口

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活conda环境
echo -e "${GREEN}激活isaaclab_env conda环境...${NC}"
eval "$(conda shell.bash hook)"
conda activate isaaclab_env

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}/source/ARM:${PYTHONPATH}"

# 显示帮助信息
show_help() {
    echo "用法: $0 [任务类型] [选项]"
    echo ""
    echo "任务类型:"
    echo "  reach       - ARM-T Reach任务（关节控制）"
    echo "  reach-ik    - ARM-T Reach任务（IK控制）"
    echo "  lift        - ARM-T Lift任务（关节控制）"
    echo "  lift-ik     - ARM-T Lift任务（IK控制）"
    echo ""
    echo "选项:"
    echo "  --headless              无头模式运行"
    echo "  --num_envs N            环境数量（默认：2048）"
    echo "  --max_iterations N      最大迭代次数"
    echo "  --wandb_project NAME    WandB项目名称"
    echo "  --wandb_entity NAME     WandB实体（用户名/团队）"
    echo "  --seed N                随机种子"
    echo ""
    echo "示例:"
    echo "  $0 reach --headless"
    echo "  $0 reach-ik --num_envs 4096 --max_iterations 2000"
    echo "  $0 lift --wandb_project my_project --wandb_entity my_team"
}

# 检查参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

TASK_TYPE=$1
shift

# 设置默认参数
HEADLESS=""
EXTRA_ARGS=""

# 解析剩余参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 记录训练开始时间（用于后续识别本次训练产生的文件）
TRAINING_START_TIME=$(date +%s)
TRAINING_DATE=$(date +%Y-%m-%d)

# 根据任务类型选择脚本和配置
case "$TASK_TYPE" in
    reach)
        echo -e "${GREEN}启动ARM-T Reach任务（关节控制）训练 - WandB集成${NC}"
        LOG_SUBDIR="arm_t_reach"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_reach.py \
            --task ARM-T-Reach-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        TRAIN_EXIT_CODE=$?
        ;;
    
    reach-ik|ik)
        echo -e "${GREEN}启动ARM-T Reach-IK任务训练 - WandB集成${NC}"
        LOG_SUBDIR="arm_t_reach_ik"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_reach_ik.py \
            --task ARM-T-Reach-IK-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach_ik.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        TRAIN_EXIT_CODE=$?
        ;;
    
    lift)
        echo -e "${GREEN}启动ARM-T Lift任务（关节控制）训练 - WandB集成${NC}"
        LOG_SUBDIR="arm_t_lift"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_lift.py \
            --task ARM-T-Lift-Cube-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        TRAIN_EXIT_CODE=$?
        ;;
    
    lift-ik)
        echo -e "${GREEN}启动ARM-T Lift任务（IK控制）训练 - WandB集成${NC}"
        LOG_SUBDIR="arm_t_lift_ik"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_lift.py \
            --task ARM-T-Lift-Cube-IK-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift_ik.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        TRAIN_EXIT_CODE=$?
        ;;
    
    *)
        echo -e "${RED}错误：未知的任务类型 '$TASK_TYPE'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}训练完成！${NC}"
else
    echo -e "${RED}训练出错（退出码: $TRAIN_EXIT_CODE）${NC}"
fi

# 查找本次训练产生的文件
echo ""
echo "=========================================="
echo "清理训练文件"
echo "=========================================="

# 查找在训练开始后创建的日志目录
TRAIN_LOGS_DIRS=()
if [ -d "${PROJECT_ROOT}/logs/rsl_rl/${LOG_SUBDIR}" ]; then
    while IFS= read -r -d '' dir; do
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
        if [ "$DIR_MTIME" -ge "$TRAINING_START_TIME" ]; then
            TRAIN_LOGS_DIRS+=("$dir")
        fi
    done < <(find "${PROJECT_ROOT}/logs/rsl_rl/${LOG_SUBDIR}" -mindepth 1 -maxdepth 1 -type d -print0)
fi

# 查找在训练开始后创建的输出目录
TRAIN_OUTPUT_DIRS=()
if [ -d "${PROJECT_ROOT}/outputs/${TRAINING_DATE}" ]; then
    while IFS= read -r -d '' dir; do
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
        if [ "$DIR_MTIME" -ge "$TRAINING_START_TIME" ]; then
            TRAIN_OUTPUT_DIRS+=("$dir")
        fi
    done < <(find "${PROJECT_ROOT}/outputs/${TRAINING_DATE}" -mindepth 1 -maxdepth 1 -type d -print0)
fi

# 显示找到的文件
TOTAL_SIZE=0
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
    echo -e "${YELLOW}是否保留本次训练内容？${NC}"
    echo "  输入 'y' 保留，其他任意键删除"
    read -p "请选择 [y/N]: " -n 1 -r KEEP_TRAINING
    echo ""
    
    if [[ ! $KEEP_TRAINING =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${YELLOW}正在删除本次训练文件...${NC}"
        
        for dir in "${TRAIN_LOGS_DIRS[@]}"; do
            echo "  删除: $dir"
            trash-put "$dir"
        done
        
        for dir in "${TRAIN_OUTPUT_DIRS[@]}"; do
            echo "  删除: $dir"
            trash-put "$dir"
        done
        
        echo -e "${GREEN}✓ 训练文件已删除${NC}"
        echo ""
        exit 0
    else
        echo -e "${GREEN}✓ 训练文件已保留${NC}"
        echo ""
        echo "日志位置: logs/rsl_rl/${LOG_SUBDIR}/"
        echo "输出位置: outputs/${TRAINING_DATE}/"
        echo ""
        echo "查看训练结果:"
        echo "  tensorboard --logdir logs/rsl_rl/${LOG_SUBDIR}"
        echo ""
    fi
else
    echo "未找到本次训练产生的文件"
fi

echo ""


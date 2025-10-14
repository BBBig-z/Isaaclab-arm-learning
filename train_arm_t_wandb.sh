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

# 根据任务类型选择脚本和配置
case "$TASK_TYPE" in
    reach)
        echo -e "${GREEN}启动ARM-T Reach任务（关节控制）训练 - WandB集成${NC}"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_reach.py \
            --task ARM-T-Reach-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        ;;
    
    reach-ik)
        echo -e "${GREEN}启动ARM-T Reach-IK任务训练 - WandB集成${NC}"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_reach_ik.py \
            --task ARM-T-Reach-IK-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach_ik.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        ;;
    
    lift)
        echo -e "${GREEN}启动ARM-T Lift任务（关节控制）训练 - WandB集成${NC}"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_lift.py \
            --task ARM-T-Lift-Cube-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        ;;
    
    lift-ik)
        echo -e "${GREEN}启动ARM-T Lift任务（IK控制）训练 - WandB集成${NC}"
        python ${PROJECT_ROOT}/scripts/rsl_rl/train_wandb_lift.py \
            --task ARM-T-Lift-Cube-IK-v0 \
            --config ${PROJECT_ROOT}/source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift_ik.yaml \
            ${HEADLESS} \
            ${EXTRA_ARGS}
        ;;
    
    *)
        echo -e "${RED}错误：未知的任务类型 '$TASK_TYPE'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

echo -e "${GREEN}训练完成！${NC}"


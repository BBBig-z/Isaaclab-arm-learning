#!/usr/bin/env python3
"""
分析ARM-T可达位姿数据库（增强版）

该脚本用于分析和可视化可达位姿数据库的内容，
帮助理解机械臂的工作空间和位姿分布。

新增功能：
- 姿态密度分析（空间分布均匀性）
- 姿态多样性评估（四元数分布）
- 3D密度热图可视化
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist


def load_database(database_path):
    """加载位姿数据库"""
    with open(database_path, 'rb') as f:
        database = pickle.load(f)
    return database


def compute_pose_density(positions, grid_size=20):
    """计算位姿密度分布（3D网格）"""
    # 创建3D网格
    x_bins = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_size)
    y_bins = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_size)
    z_bins = np.linspace(positions[:, 2].min(), positions[:, 2].max(), grid_size)
    
    # 计算每个网格单元的位姿数量
    hist, edges = np.histogramdd(positions, bins=[x_bins, y_bins, z_bins])
    
    # 计算密度统计
    total_cells = hist.size
    occupied_cells = np.count_nonzero(hist)
    occupation_ratio = occupied_cells / total_cells
    
    # 密度均匀性（使用标准差/均值）
    non_zero_densities = hist[hist > 0]
    if len(non_zero_densities) > 0:
        uniformity = 1 - (non_zero_densities.std() / (non_zero_densities.mean() + 1e-6))
    else:
        uniformity = 0
    
    return {
        'histogram': hist,
        'edges': edges,
        'total_cells': total_cells,
        'occupied_cells': occupied_cells,
        'occupation_ratio': occupation_ratio,
        'uniformity': max(0, uniformity),  # 0-1范围，越大越均匀
        'max_density': hist.max(),
        'mean_density': non_zero_densities.mean() if len(non_zero_densities) > 0 else 0,
    }


def compute_orientation_diversity(orientations_quat):
    """计算姿态多样性（四元数角度距离）"""
    # 采样部分四元数计算平均角度距离（全量计算太慢）
    sample_size = min(1000, len(orientations_quat))
    sample_indices = np.random.choice(len(orientations_quat), sample_size, replace=False)
    sampled_quats = orientations_quat[sample_indices]
    
    # 计算四元数之间的角度距离
    # d(q1, q2) = 2 * arccos(|q1 · q2|)
    angles = []
    for i in range(len(sampled_quats)):
        for j in range(i+1, len(sampled_quats)):
            dot_product = np.abs(np.dot(sampled_quats[i], sampled_quats[j]))
            dot_product = np.clip(dot_product, -1.0, 1.0)  # 数值稳定性
            angle = 2 * np.arccos(dot_product)
            angles.append(np.degrees(angle))
    
    angles = np.array(angles)
    
    return {
        'mean_angle_diff': angles.mean(),
        'std_angle_diff': angles.std(),
        'min_angle_diff': angles.min(),
        'max_angle_diff': angles.max(),
    }


def analyze_database(database):
    """分析数据库统计信息（增强版）"""
    print("=" * 70)
    print("ARM-T 可达位姿数据库分析（增强版）")
    print("=" * 70)
    print()
    
    print(f"总位姿数量: {database['num_poses']}")
    print(f"生成方法: {database.get('generation_method', 'unknown')}")
    
    # 碰撞检测统计
    if 'collision_check_enabled' in database:
        print(f"碰撞检测: {'✓ 已启用' if database['collision_check_enabled'] else '✗ 未启用'}")
    if 'statistics' in database:
        stats = database['statistics']
        print(f"  总尝试次数: {stats['total_attempts']}")
        print(f"  碰撞拒绝: {stats['collision_rejected']}")
        print(f"  工作空间过滤: {stats['workspace_filtered']}")
        print(f"  成功率: {stats['success_rate']*100:.1f}%")
    print()
    
    positions = database['positions']
    orientations = database['orientations_quat']
    
    # 位置统计
    print("【位置统计】(米)")
    print(f"  X轴: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"均值={positions[:, 0].mean():.3f}, 标准差={positions[:, 0].std():.3f}")
    print(f"  Y轴: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"均值={positions[:, 1].mean():.3f}, 标准差={positions[:, 1].std():.3f}")
    print(f"  Z轴: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}], "
          f"均值={positions[:, 2].mean():.3f}, 标准差={positions[:, 2].std():.3f}")
    print()
    
    # 工作空间体积估计
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    z_range = positions[:, 2].max() - positions[:, 2].min()
    volume = x_range * y_range * z_range
    print(f"工作空间估计体积: {volume:.3f} 立方米")
    print()
    
    # 姿态密度分析
    print("【姿态密度分析】")
    density_info = compute_pose_density(positions, grid_size=15)
    print(f"  网格单元总数: {density_info['total_cells']}")
    print(f"  占用单元数: {density_info['occupied_cells']}")
    print(f"  空间占用率: {density_info['occupation_ratio']*100:.1f}%")
    print(f"  最大密度: {int(density_info['max_density'])} 位姿/单元")
    print(f"  平均密度: {density_info['mean_density']:.1f} 位姿/单元")
    print(f"  分布均匀性: {density_info['uniformity']*100:.1f}% (0%=极不均匀, 100%=完全均匀)")
    print()
    
    # 姿态多样性分析
    print("【姿态多样性分析】")
    try:
        orientation_diversity = compute_orientation_diversity(orientations)
        print(f"  平均角度差异: {orientation_diversity['mean_angle_diff']:.1f}°")
        print(f"  角度差异标准差: {orientation_diversity['std_angle_diff']:.1f}°")
        print(f"  最小角度差异: {orientation_diversity['min_angle_diff']:.1f}°")
        print(f"  最大角度差异: {orientation_diversity['max_angle_diff']:.1f}°")
        
        # 多样性评估
        if orientation_diversity['mean_angle_diff'] > 60:
            diversity_level = "高"
        elif orientation_diversity['mean_angle_diff'] > 30:
            diversity_level = "中等"
        else:
            diversity_level = "低"
        print(f"  多样性等级: {diversity_level}")
    except Exception as e:
        print(f"  无法计算姿态多样性: {e}")
    print()
    
    # 姿态统计（四元数）
    print("【姿态统计】(四元数)")
    print(f"  Qw: [{orientations[:, 0].min():.3f}, {orientations[:, 0].max():.3f}], "
          f"均值={orientations[:, 0].mean():.3f}")
    print(f"  Qx: [{orientations[:, 1].min():.3f}, {orientations[:, 1].max():.3f}], "
          f"均值={orientations[:, 1].mean():.3f}")
    print(f"  Qy: [{orientations[:, 2].min():.3f}, {orientations[:, 2].max():.3f}], "
          f"均值={orientations[:, 2].mean():.3f}")
    print(f"  Qz: [{orientations[:, 3].min():.3f}, {orientations[:, 3].max():.3f}], "
          f"均值={orientations[:, 3].mean():.3f}")
    print()


def visualize_workspace(database, output_path=None):
    """可视化工作空间（增强版 - 包含密度热图）"""
    positions = database['positions']
    
    # 计算密度
    density_info = compute_pose_density(positions, grid_size=15)
    
    # 创建大图
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D散点图
    ax1 = fig.add_subplot(231, projection='3d')
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=positions[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D工作空间')
    plt.colorbar(scatter, ax=ax1, label='Z高度 (m)', shrink=0.5)
    
    # 2. XY平面密度热图
    ax2 = fig.add_subplot(232)
    xy_density = density_info['histogram'].sum(axis=2)  # 沿Z轴求和
    im2 = ax2.imshow(xy_density.T, origin='lower', cmap='hot', aspect='auto',
                     extent=[positions[:, 0].min(), positions[:, 0].max(),
                            positions[:, 1].min(), positions[:, 1].max()])
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面密度热图')
    plt.colorbar(im2, ax=ax2, label='位姿数量')
    
    # 3. XZ平面密度热图
    ax3 = fig.add_subplot(233)
    xz_density = density_info['histogram'].sum(axis=1)  # 沿Y轴求和
    im3 = ax3.imshow(xz_density.T, origin='lower', cmap='hot', aspect='auto',
                     extent=[positions[:, 0].min(), positions[:, 0].max(),
                            positions[:, 2].min(), positions[:, 2].max()])
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ平面密度热图')
    plt.colorbar(im3, ax=ax3, label='位姿数量')
    
    # 4. YZ平面密度热图
    ax4 = fig.add_subplot(234)
    yz_density = density_info['histogram'].sum(axis=0)  # 沿X轴求和
    im4 = ax4.imshow(yz_density.T, origin='lower', cmap='hot', aspect='auto',
                     extent=[positions[:, 1].min(), positions[:, 1].max(),
                            positions[:, 2].min(), positions[:, 2].max()])
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('YZ平面密度热图')
    plt.colorbar(im4, ax=ax4, label='位姿数量')
    
    # 5. 密度分布直方图
    ax5 = fig.add_subplot(235)
    non_zero_densities = density_info['histogram'][density_info['histogram'] > 0]
    ax5.hist(non_zero_densities, bins=30, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('单元密度（位姿数量）')
    ax5.set_ylabel('频数')
    ax5.set_title('密度分布直方图')
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计信息文本
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    stats_text = f"""
数据库统计信息

总位姿数: {len(positions)}
工作空间体积: {(positions[:, 0].max() - positions[:, 0].min()) * 
               (positions[:, 1].max() - positions[:, 1].min()) * 
               (positions[:, 2].max() - positions[:, 2].min()):.3f} m³

密度分析:
  空间占用率: {density_info['occupation_ratio']*100:.1f}%
  分布均匀性: {density_info['uniformity']*100:.1f}%
  最大密度: {int(density_info['max_density'])} 位姿/单元
  平均密度: {density_info['mean_density']:.1f} 位姿/单元

X范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m
Y范围: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m
Z范围: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存到: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="分析ARM-T可达位姿数据库")
    parser.add_argument("--database", type=str, 
                       default="source/ARM/arm_t/tasks/reach/reachable_poses_database.pkl",
                       help="数据库文件路径")
    parser.add_argument("--visualize", action="store_true", help="可视化工作空间")
    parser.add_argument("--output", type=str, default=None, help="可视化输出路径")
    
    args = parser.parse_args()
    
    database_path = Path(args.database)
    
    if not database_path.exists():
        print(f"错误: 数据库文件不存在: {database_path}")
        print("请先运行 generate_reachable_poses.py 生成数据库")
        return
    
    # 加载并分析数据库
    database = load_database(database_path)
    analyze_database(database)
    
    # 可视化（如果需要）
    if args.visualize:
        try:
            visualize_workspace(database, args.output)
        except ImportError:
            print("警告: 无法导入matplotlib，跳过可视化")
            print("请安装: pip install matplotlib")


if __name__ == "__main__":
    main()


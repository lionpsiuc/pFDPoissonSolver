import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_poisson2d_solution(filename=None, show_error=True):
    """
    读取并绘制2D Poisson方程的解决方案。

    参数:
    -----
    filename : str
        包含解决方案的文件名，未指定则使用命令行参数或默认值。
    show_error : bool
        是否计算并显示与解析解的误差。
    """
    # 确定文件名
    if filename is None:
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        else:
            # 默认使用4进程解
            filename = "poisson2d_sol_nx31_np4"

    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误: 文件{filename}不存在!")
        print("请确保已运行Q4代码生成解决方案文件。")
        return

    print(f"读取文件: {filename}")

    # 提取nx值和进程数
    import re
    match = re.search(r'nx(\d+)_np(\d+)', filename)
    if match:
        nx = int(match.group(1))
        nprocs = int(match.group(2))
    else:
        nx = 31  # 默认值
        nprocs = 4  # 默认值

    # 读取数据
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 处理头部注释
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            start_idx = i + 1
        else:
            break

    # 解析数据
    x_vals = []
    y_vals = []
    z_vals = []

    for line in lines[start_idx:]:
        # 跳过空行
        if not line.strip():
            continue

        # 解析数据行
        try:
            values = line.strip().split()
            if len(values) >= 3:
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
        except ValueError:
            continue

    # 转换为NumPy数组
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    # 确定网格尺寸
    unique_x = np.unique(x_vals)
    unique_y = np.unique(y_vals)
    nx_actual = len(unique_x) - 1  # 减去边界点
    ny_actual = len(unique_y) - 1  # 减去边界点

    print(f"网格尺寸: {nx_actual}x{ny_actual}")

    # 创建网格
    x_grid, y_grid = np.meshgrid(unique_x, unique_y)
    z_grid = np.zeros_like(x_grid)

    # 填充网格值
    for i, y in enumerate(unique_y):
        for j, x in enumerate(unique_x):
            # 查找对应的值
            idx = np.where((x_vals == x) & (y_vals == y))[0]
            if len(idx) > 0:
                z_grid[i, j] = z_vals[idx[0]]

    # 计算误差
    if show_error:
        h = 1.0 / (nx + 1)
        analytical_solution = np.zeros_like(z_grid)
        max_error = 0.0

        for i, y in enumerate(unique_y):
            for j, x in enumerate(unique_x):
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # 解析解: u(x,y) = y/((1+x)^2 + y^2)
                    analytical = y / ((1.0 + x)**2 + y**2) if (y != 0 or x != -1) else 0
                    analytical_solution[i, j] = analytical
                    error = abs(z_grid[i, j] - analytical)
                    if error > max_error:
                        max_error = error

        print(f"最大误差: {max_error:.6e}")

    # 创建图表
    plt.figure(figsize=(12, 10))

    # 绘制热图
    plt.subplot(2, 2, 1)
    contour = plt.contourf(x_grid, y_grid, z_grid, 50, cmap=cm.viridis)
    plt.colorbar(label='u(x,y)')
    plt.title(f'Poisson方程解 - 热图 (nx={nx}, np={nprocs})')
    plt.xlabel('x')
    plt.ylabel('y')

    # 绘制3D表面图
    ax = plt.subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.viridis, edgecolor='none')
    ax.set_title(f'Poisson方程解 - 3D表面 (nx={nx}, np={nprocs})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')

    if show_error:
        # 绘制误差热图
        plt.subplot(2, 2, 3)
        error_grid = np.abs(z_grid - analytical_solution)
        error_contour = plt.contourf(x_grid, y_grid, error_grid, 50, cmap=cm.hot)
        plt.colorbar(label='Error')
        plt.title(f'误差分布 (最大误差: {max_error:.6e})')
        plt.xlabel('x')
        plt.ylabel('y')

        # 绘制解析解
        plt.subplot(2, 2, 4)
        analytical_contour = plt.contourf(x_grid, y_grid, analytical_solution, 50, cmap=cm.viridis)
        plt.colorbar(label='u_exact(x,y)')
        plt.title('解析解')
        plt.xlabel('x')
        plt.ylabel('y')

    plt.tight_layout()

    # 保存图像
    output_file = f"{filename}_visualization.png"
    plt.savefig(output_file, dpi=300)
    print(f"图像已保存至: {output_file}")

    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 如果提供了文件名参数，则使用该文件
    if len(sys.argv) > 1:
        plot_poisson2d_solution(sys.argv[1])
    else:
        # 否则，使用默认值
        plot_poisson2d_solution()

        # 自动绘制16进程的结果
        if os.path.exists("poisson2d_sol_nx31_np16"):
            plot_poisson2d_solution("poisson2d_sol_nx31_np16")

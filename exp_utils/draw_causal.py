import matplotlib.pyplot as plt
import numpy as np

# 1. 定义所有相关数据
# 我们只关注 deterministic = True 和 Fav3 bwd 的性能
seq_labels = ['4096', '8192', '16384']
x = np.arange(len(seq_labels))  # x轴位置

# --- 优化后 (新数据) ---
opt_causal = {
    'bs2': [398.2, 450.7, 465.1],
    'bs1': [376.6, 447.7, 473.7]
}
opt_non_causal = {
    'bs2': [512.6, 551.2, 550.0],
    'bs1': [501.1, 558.5, 553.8]
}

# --- 优化前 (旧数据) ---
unopt_causal = {
    'bs2': [371.3, 419.5, 444.7],
    'bs1': [340.8, 406.6, 438.8]
}
unopt_non_causal = {
    'bs2': [401.7, 485.2, 549.7],
    'bs1': [359.4, 458.5, 531.5]
}

# 2. 计算加速比 (Speedup)
speedup_causal_bs2 = np.array(opt_causal['bs2']) / np.array(unopt_causal['bs2'])
speedup_causal_bs1 = np.array(opt_causal['bs1']) / np.array(unopt_causal['bs1'])
speedup_non_causal_bs2 = np.array(opt_non_causal['bs2']) / np.array(unopt_non_causal['bs2'])
speedup_non_causal_bs1 = np.array(opt_non_causal['bs1']) / np.array(unopt_non_causal['bs1'])

# 3. 创建可视化图表 (使用子图)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16))
fig.suptitle('Fav3 BWD 性能与加速比对比 (Deterministic)', fontsize=20, y=0.95)

bar_width = 0.2  # 柱形宽度

# --- 绘制子图 1: Batch Size = 2 ---
ax1.set_title('Batch Size = 2', fontsize=16)
# 绘制 Causal 对比
br1 = ax1.bar(x - 1.5 * bar_width, unopt_causal['bs2'], width=bar_width, label='Causal (优化前)', color='lightcoral')
br2 = ax1.bar(x - 0.5 * bar_width, opt_causal['bs2'], width=bar_width, label='Causal (优化后)', color='crimson')
# 绘制 Non-Causal 对比
br3 = ax1.bar(x + 0.5 * bar_width, unopt_non_causal['bs2'], width=bar_width, label='Non-Causal (优化前)', color='skyblue')
br4 = ax1.bar(x + 1.5 * bar_width, opt_non_causal['bs2'], width=bar_width, label='Non-Causal (优化后)', color='royalblue')

# --- 绘制子图 2: Batch Size = 1 ---
ax2.set_title('Batch Size = 1', fontsize=16)
# 绘制 Causal 对比
br5 = ax2.bar(x - 1.5 * bar_width, unopt_causal['bs1'], width=bar_width, label='Causal (优化前)', color='lightcoral')
br6 = ax2.bar(x - 0.5 * bar_width, opt_causal['bs1'], width=bar_width, label='Causal (优化后)', color='crimson')
# 绘制 Non-Causal 对比
br7 = ax2.bar(x + 0.5 * bar_width, unopt_non_causal['bs1'], width=bar_width, label='Non-Causal (优化前)', color='skyblue')
br8 = ax2.bar(x + 1.5 * bar_width, opt_non_causal['bs1'], width=bar_width, label='Non-Causal (优化后)', color='royalblue')


# 4. 定义一个函数来添加标签，避免代码重复
def add_labels(ax, bars, speedups=None):
    """为柱形添加 TFLOPS 值和可选的加速比"""
    # 添加 TFLOPS 数值标签
    ax.bar_label(bars, padding=3, fmt='%.1f', fontsize=10)
    
    # 如果提供了 speedup，则添加加速比标签
    if speedups is not None:
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            # 在数值标签上方添加加速比
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 18, f'{speedups[i]:.2f}x',
                    ha='center', va='bottom', color='black', weight='bold', fontsize=11,
                    bbox=dict(facecolor='gold', alpha=0.6, boxstyle='round,pad=0.2'))

# 5. 为所有柱形和子图应用标签和样式
# 应用到子图1 (BS=2)
add_labels(ax1, br1)
add_labels(ax1, br2, speedup_causal_bs2)
add_labels(ax1, br3)
add_labels(ax1, br4, speedup_non_causal_bs2)

# 应用到子图2 (BS=1)
add_labels(ax2, br5)
add_labels(ax2, br6, speedup_causal_bs1)
add_labels(ax2, br7)
add_labels(ax2, br8, speedup_non_causal_bs1)

# 统一设置子图样式
for ax in [ax1, ax2]:
    ax.set_ylabel('性能 (TFLOPS)', fontsize=14)
    ax.set_xlabel('序列长度 (Sequence Length)', fontsize=14, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='upper left')
    
    # 调整Y轴范围以容纳所有标签
    y_max = max([b.get_height() for b in ax.patches])
    ax.set_ylim(0, y_max * 1.2)


# 6. 显示图表
fig.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局以适应主标题
plt.savefig('fav3_bwd_performance_comparison.png', dpi=300)
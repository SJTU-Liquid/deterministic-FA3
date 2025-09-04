import matplotlib.pyplot as plt
import numpy as np

# 1. 定义数据
# 筛选出 causal=True, deterministic=True, Fav3 bwd 的数据
seq_lengths_labels = ['4096', '8192', '16384']
seq_lengths = np.arange(len(seq_lengths_labels))

# 优化后的 TFLOPS 数据
optimized = {
    'bs2': [393.8, 438.0, 460.2],
    'bs1': [372.6, 437.4, 458.6]
}

# 优化前的 TFLOPS 数据
unoptimized = {
    'bs2': [371.3, 419.5, 444.7],
    'bs1': [340.8, 406.6, 438.8]
}

# 2. 计算加速比 (Speedup)
# Speedup = Optimized TFLOPS / Unoptimized TFLOPS
speedup_bs2 = np.array(optimized['bs2']) / np.array(unoptimized['bs2'])
speedup_bs1 = np.array(optimized['bs1']) / np.array(unoptimized['bs1'])

# 3. 创建可视化图表

# 设置图表大小
fig, ax = plt.subplots(figsize=(16, 9))

# 设置柱形的宽度
bar_width = 0.2

# 绘制 Batch Size = 2 的柱形
br1 = ax.bar(seq_lengths - 1.5 * bar_width, unoptimized['bs2'], width=bar_width, label='优化前 (BS=2)', color='lightcoral')
br2 = ax.bar(seq_lengths - 0.5 * bar_width, optimized['bs2'], width=bar_width, label='优化后 (BS=2)', color='crimson')

# 绘制 Batch Size = 1 的柱形
br3 = ax.bar(seq_lengths + 0.5 * bar_width, unoptimized['bs1'], width=bar_width, label='优化前 (BS=1)', color='skyblue')
br4 = ax.bar(seq_lengths + 1.5 * bar_width, optimized['bs1'], width=bar_width, label='优化后 (BS=1)', color='royalblue')

# 4. 添加图表元素和样式

# 添加标题和坐标轴标签
ax.set_title('Fav3 BWD 性能与加速比 (Causal & Deterministic)', fontsize=18, pad=20)
ax.set_xlabel('序列长度 (Sequence Length)', fontsize=14, labelpad=15)
ax.set_ylabel('性能 (TFLOPS)', fontsize=14)
ax.set_xticks(seq_lengths)
ax.set_xticklabels(seq_lengths_labels)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=12)

# 5. 为柱形添加数值和加速比标签

# 添加 TFLOPS 数值标签
ax.bar_label(br1, padding=3, fmt='%.1f')
ax.bar_label(br2, padding=3, fmt='%.1f')
ax.bar_label(br3, padding=3, fmt='%.1f')
ax.bar_label(br4, padding=3, fmt='%.1f')

# 在“优化后”的柱形上方添加加速比标签
for i, bar in enumerate(br2):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{speedup_bs2[i]:.2f}x', 
            ha='center', va='bottom', color='black', weight='bold', fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

for i, bar in enumerate(br4):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{speedup_bs1[i]:.2f}x', 
            ha='center', va='bottom', color='black', weight='bold', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

# 调整 Y 轴范围以容纳标签
ax.set_ylim(0, max(max(optimized['bs2']), max(optimized['bs1'])) * 1.15)


# 调整布局
fig.tight_layout()

# 6. 显示图表
plt.savefig("fav3_bwd_causal_deterministic.png")
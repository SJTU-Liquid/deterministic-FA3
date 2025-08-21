import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# 优化前的数据
data_before_str = """
### headdim = 128, causal = False, seqlen = 16384, deterministic = False ###
Fav2 fwd: 12.252ms, 359.0 TFLOPS
Fav2 bwd: 35.758ms, 307.5 TFLOPS
Fav3 fwd: 5.837ms, 753.5 TFLOPS
Fav3 bwd: 18.034ms, 609.7 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
Fav2 fwd: 12.308ms, 357.3 TFLOPS
Fav2 bwd: 60.364ms, 182.1 TFLOPS
Fav3 fwd: 5.830ms, 754.4 TFLOPS
Fav3 bwd: 18.884ms, 582.3 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = False ###
Fav2 fwd: 3.381ms, 325.2 TFLOPS
Fav2 bwd: 9.087ms, 302.5 TFLOPS
Fav3 fwd: 1.480ms, 743.1 TFLOPS
Fav3 bwd: 4.278ms, 642.5 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = True ###
Fav2 fwd: 3.455ms, 318.3 TFLOPS
Fav2 bwd: 15.327ms, 179.3 TFLOPS
Fav3 fwd: 1.479ms, 743.3 TFLOPS
Fav3 bwd: 5.015ms, 548.1 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = False ###
Fav2 fwd: 0.844ms, 325.7 TFLOPS
Fav2 bwd: 2.397ms, 286.7 TFLOPS
Fav3 fwd: 0.390ms, 705.7 TFLOPS
Fav3 bwd: 1.134ms, 605.7 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = True ###
Fav2 fwd: 0.774ms, 354.9 TFLOPS
Fav2 bwd: 3.965ms, 173.3 TFLOPS
Fav3 fwd: 0.389ms, 705.8 TFLOPS
Fav3 bwd: 1.467ms, 468.5 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = False ###
Fav2 fwd: 6.005ms, 366.2 TFLOPS
Fav2 bwd: 17.852ms, 308.0 TFLOPS
Fav3 fwd: 2.956ms, 744.0 TFLOPS
Fav3 bwd: 8.423ms, 652.7 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
Fav2 fwd: 6.335ms, 347.1 TFLOPS
Fav2 bwd: 34.411ms, 159.8 TFLOPS
Fav3 fwd: 2.930ms, 750.4 TFLOPS
Fav3 bwd: 9.387ms, 585.7 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = False ###
Fav2 fwd: 1.745ms, 315.0 TFLOPS
Fav2 bwd: 4.600ms, 298.8 TFLOPS
Fav3 fwd: 0.746ms, 736.6 TFLOPS
Fav3 bwd: 2.109ms, 651.6 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = True ###
Fav2 fwd: 1.630ms, 337.3 TFLOPS
Fav2 bwd: 8.890ms, 154.6 TFLOPS
Fav3 fwd: 0.746ms, 736.6 TFLOPS
Fav3 bwd: 2.617ms, 525.1 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = False ###
Fav2 fwd: 0.397ms, 346.5 TFLOPS
Fav2 bwd: 1.238ms, 277.4 TFLOPS
Fav3 fwd: 0.202ms, 681.9 TFLOPS
Fav3 bwd: 0.589ms, 583.5 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = True ###
Fav2 fwd: 0.397ms, 345.9 TFLOPS
Fav2 bwd: 2.268ms, 151.5 TFLOPS
Fav3 fwd: 0.201ms, 682.3 TFLOPS
Fav3 bwd: 0.827ms, 415.5 TFLOPS
"""

# 优化后的数据
data_after_str = """
### headdim = 128, causal = False, seqlen = 16384, deterministic = False ###
Fav2 fwd: 12.150ms, 362.0 TFLOPS
Fav2 bwd: 35.796ms, 307.2 TFLOPS
Fav3 fwd: 6.214ms, 707.8 TFLOPS
Fav3 bwd: 18.357ms, 598.9 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
Fav2 fwd: 12.353ms, 356.0 TFLOPS
Fav2 bwd: 60.292ms, 182.4 TFLOPS
Fav3 fwd: 5.912ms, 743.9 TFLOPS
Fav3 bwd: 20.017ms, 549.3 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = False ###
Fav2 fwd: 3.228ms, 340.6 TFLOPS
Fav2 bwd: 9.099ms, 302.1 TFLOPS
Fav3 fwd: 1.493ms, 736.4 TFLOPS
Fav3 bwd: 4.369ms, 629.2 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = True ###
Fav2 fwd: 3.548ms, 309.9 TFLOPS
Fav2 bwd: 15.201ms, 180.8 TFLOPS
Fav3 fwd: 1.492ms, 736.7 TFLOPS
Fav3 bwd: 4.930ms, 557.5 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = False ###
Fav2 fwd: 0.833ms, 329.8 TFLOPS
Fav2 bwd: 2.405ms, 285.8 TFLOPS
Fav3 fwd: 0.391ms, 702.4 TFLOPS
Fav3 bwd: 1.161ms, 592.0 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = True ###
Fav2 fwd: 0.776ms, 354.4 TFLOPS
Fav2 bwd: 3.990ms, 172.2 TFLOPS
Fav3 fwd: 0.391ms, 703.6 TFLOPS
Fav3 bwd: 1.334ms, 515.2 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = False ###
Fav2 fwd: 6.052ms, 363.3 TFLOPS
Fav2 bwd: 17.936ms, 306.5 TFLOPS
Fav3 fwd: 2.957ms, 743.8 TFLOPS
Fav3 bwd: 8.836ms, 622.2 TFLOPS

### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
Fav2 fwd: 6.436ms, 341.7 TFLOPS
Fav2 bwd: 34.710ms, 158.4 TFLOPS
Fav3 fwd: 2.982ms, 737.4 TFLOPS
Fav3 bwd: 9.878ms, 556.5 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = False ###
Fav2 fwd: 1.661ms, 331.0 TFLOPS
Fav2 bwd: 4.599ms, 298.8 TFLOPS
Fav3 fwd: 0.751ms, 732.3 TFLOPS
Fav3 bwd: 2.146ms, 640.4 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = True ###
Fav2 fwd: 1.675ms, 328.3 TFLOPS
Fav2 bwd: 8.887ms, 154.7 TFLOPS
Fav3 fwd: 0.751ms, 732.4 TFLOPS
Fav3 bwd: 2.475ms, 555.4 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = False ###
Fav2 fwd: 0.414ms, 331.6 TFLOPS
Fav2 bwd: 1.254ms, 274.0 TFLOPS
Fav3 fwd: 0.202ms, 679.9 TFLOPS
Fav3 bwd: 0.607ms, 565.9 TFLOPS

### headdim = 128, causal = False, seqlen = 4096, deterministic = True ###
Fav2 fwd: 0.395ms, 347.5 TFLOPS
Fav2 bwd: 2.287ms, 150.2 TFLOPS
Fav3 fwd: 0.202ms, 680.7 TFLOPS
Fav3 bwd: 0.693ms, 495.8 TFLOPS
"""

def parse_data(data_str, status):
    """
    解析函数，为每个###块添加一个唯一的运行索引(Run Index)。
    """
    line_regex = re.compile(r"(Fav\d)\s+(fwd|bwd):\s+([\d.]+)\s*ms,\s+([\d.]+)\s*TFLOPS")
    blocks = re.findall(r'###(.*?)###\n(.*?)(?=\n###|\Z)', data_str, re.DOTALL)
    
    parsed_records = []

    # --- 修改点: 使用 enumerate 为每个 ### 块创建唯一索引 ---
    for i, (params_str, results_str) in enumerate(blocks):
        test_condition_label = params_str.strip()
        seqlen_match = re.search(r"seqlen = (\d+)", params_str)
        deterministic_match = re.search(r"deterministic = (True|False)", params_str)

        seqlen = int(seqlen_match.group(1)) if seqlen_match else None
        deterministic = deterministic_match.group(1) == 'True' if deterministic_match else None
        
        for line in results_str.strip().split('\n'):
            match = line_regex.search(line)
            if match:
                kernel, direction, time_ms, tflops = match.groups()
                parsed_records.append({
                    'Run Index': i, # 用于区分不同测试运行的唯一ID
                    'Optimization': status,
                    'Test Condition': test_condition_label,
                    'Seqlen': seqlen,
                    'Deterministic': deterministic,
                    'Kernel': kernel,
                    'Direction': direction.upper(),
                    'Time (ms)': float(time_ms),
                    'TFLOPS': float(tflops)
                })
    return parsed_records

# 解析优化前后的数据
data_before = parse_data(data_before_str, 'Before')
data_after = parse_data(data_after_str, 'After')

# 合并成一个DataFrame
df = pd.DataFrame(data_before + data_after)

# --- 修改点: 只筛选出 Fav3, BWD, 且 Deterministic = True 的数据 ---
df_final = df[
    (df['Kernel'] == 'Fav3') &
    (df['Direction'] == 'BWD') &
    (df['Deterministic'] == True)
].copy()

# --- 修改点: 创建 Batch 标识和唯一的 X 轴标签 ---
# 数据中每6个 ### 块为一轮完整的测试，以此区分 Batch
df_final['Batch'] = 2 - ((df_final['Run Index'] // 6))
df_final['X_Label'] = df_final.apply(
    lambda row: f"Batch {row['Batch']}\n{row['Test Condition']}",
    axis=1
)

# 根据 Batch 和 Seqlen (降序) 对X轴进行排序
sorted_x_labels = df_final.sort_values(
    by=['Batch', 'Seqlen'],
    ascending=[True, False]
)['X_Label'].unique()

# 定义图例的顺序
hue_order = ['Before', 'After']

# --- 修改点: 更新绘图逻辑 ---
sns.set_theme(style="whitegrid")

# 图1: TFLOPS 性能对比图
plt.figure(figsize=(16, 9))
g_tflops = sns.barplot(
    data=df_final,
    x='X_Label',
    y='TFLOPS',
    hue='Optimization', # Hue 现在只区分优化状态
    order=sorted_x_labels,
    hue_order=hue_order,
    palette='viridis'
)
g_tflops.set_title('Fav3 Backward Pass (BWD) TFLOPS Comparison (Deterministic Only)', fontsize=18)
g_tflops.set_xlabel('Test Parameters', fontsize=14)
g_tflops.set_ylabel('TFLOPS (Higher is Better)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Optimization Status')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

for p in g_tflops.patches:
    g_tflops.annotate(f"{p.get_height():.1f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=9)
plt.savefig('fav3_bwd_tflops_deterministic_only.png')
print("已生成 BWD TFLOPS 对比图: fav3_bwd_tflops_deterministic_only.png")

# 图2: 执行时间(ms)对比图
plt.figure(figsize=(16, 9))
g_ms = sns.barplot(
    data=df_final,
    x='X_Label',
    y='Time (ms)',
    hue='Optimization', # Hue 现在只区分优化状态
    order=sorted_x_labels,
    hue_order=hue_order,
    palette='plasma'
)
g_ms.set_title('Fav3 Backward Pass (BWD) Execution Time Comparison (Deterministic Only)', fontsize=18)
g_ms.set_xlabel('Test Parameters', fontsize=14)
g_ms.set_ylabel('Time in ms (Lower is Better)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Optimization Status')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

for p in g_ms.patches:
    g_ms.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=9)

plt.savefig('fav3_bwd_time_deterministic_only.png')
print("已生成 BWD 执行时间对比图: fav3_bwd_time_deterministic_only.png")
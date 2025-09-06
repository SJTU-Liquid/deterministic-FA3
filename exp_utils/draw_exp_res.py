import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ==================== 1. 加载数据 ====================
# 将您提供的CSV数据加载到字符串中
baseline_data = """headdim,causal,seqlen,Fav3 bwd (ms),Fav3 bwd (TFLOPS)
64,False,512,0.708,242.8
64,True,512,0.570,150.7
64,False,1024,1.117,307.5
64,True,1024,0.706,243.4
64,False,2048,1.944,353.4
64,True,2048,1.027,334.4
64,False,4096,3.460,397.2
64,True,4096,1.746,393.7
64,False,8192,6.119,449.2
64,True,8192,3.175,432.8
64,False,16384,10.823,507.9
64,True,16384,6.103,450.4
128,False,512,0.660,260.2
128,True,512,0.536,160.4
128,False,1024,0.939,366.1
128,True,1024,0.710,241.9
128,False,2048,1.587,432.9
128,True,2048,1.103,311.4
128,False,4096,2.775,495.2
128,True,4096,1.905,360.8
128,False,8192,4.908,560.0
128,True,8192,3.538,388.5
128,False,16384,9.478,580.0
128,True,16384,6.898,398.5
"""

descending_data = """headdim,causal,seqlen,Fav3 bwd (ms),Fav3 bwd (TFLOPS)
64,False,512,0.694,247.4
64,True,512,0.582,147.6
64,False,1024,1.117,307.6
64,True,1024,0.726,236.5
64,False,2048,1.924,357.2
64,True,2048,1.042,329.8
64,False,4096,3.418,402.1
64,True,4096,1.725,398.4
64,False,8192,6.066,453.2
64,True,8192,3.108,442.2
64,False,16384,10.671,515.2
64,True,16384,5.941,462.7
128,False,512,0.653,263.0
128,True,512,0.530,162.0
128,False,1024,0.946,363.2
128,True,1024,0.679,253.1
128,False,2048,1.578,435.5
128,True,2048,0.962,357.3
128,False,4096,2.779,494.5
128,True,4096,1.577,435.6
128,False,8192,5.013,548.4
128,True,8192,2.810,489.1
128,False,16384,9.426,583.2
128,True,16384,5.372,511.6
"""

persistent_descending_data = """headdim,causal,seqlen,Fav3 bwd (ms),Fav3 bwd (TFLOPS)
64,False,512,0.605,283.9
64,True,512,0.538,159.8
64,False,1024,1.136,302.4
64,True,1024,0.725,237.1
64,False,2048,2.139,321.3
64,True,2048,1.159,296.5
64,False,4096,3.878,354.4
64,True,4096,1.951,352.2
64,False,8192,6.639,414.0
64,True,8192,3.670,374.5
64,False,16384,10.693,514.1
64,True,16384,7.171,383.3
128,False,512,0.622,276.1
128,True,512,0.553,155.3
128,False,1024,0.960,357.9
128,True,1024,0.676,254.1
128,False,2048,1.706,402.8
128,True,2048,1.011,340.0
128,False,4096,2.960,464.3
128,True,4096,1.737,395.6
128,False,8192,4.983,551.6
128,True,8192,2.965,463.5
128,False,16384,9.570,574.5
128,True,16384,5.540,496.2
"""

shift_data_persistent = """headdim,causal,seqlen,Fav3 bwd (ms),Fav3 bwd (TFLOPS)
64,False,512,0.623,275.6
64,True,512,0.537,159.9
64,False,1024,0.934,367.8
64,True,1024,0.686,250.5
64,False,2048,1.594,431.2
64,True,2048,1.020,336.9
64,False,4096,2.864,479.9
64,True,4096,1.702,403.8
64,False,8192,5.509,499.0
64,True,8192,3.110,442.0
64,False,16384,10.973,501.0
64,True,16384,6.000,458.1
128,False,512,0.647,265.7
128,True,512,0.554,155.1
128,False,1024,0.904,380.1
128,True,1024,0.690,248.8
128,False,2048,1.483,463.4
128,True,2048,1.007,341.3
128,False,4096,2.629,522.9
128,True,4096,1.678,409.6
128,False,8192,4.938,556.7
128,True,8192,3.045,451.3
128,False,16384,9.874,556.8
128,True,16384,5.806,473.5
"""

shift_data = """headdim,causal,seqlen,Fav3 bwd (ms),Fav3 bwd (TFLOPS)
64,False,512,0.652,263.4
64,True,512,0.528,162.6
64,False,1024,0.995,345.3
64,True,1024,0.701,245.1
64,False,2048,1.652,415.9
64,True,2048,1.028,334.1
64,False,4096,2.948,466.2
64,True,4096,1.720,399.5
64,False,8192,5.571,493.4
64,True,8192,3.134,438.5
64,False,16384,11.012,499.3
64,True,16384,6.010,457.3
128,False,512,0.659,260.9
128,True,512,0.536,160.4
128,False,1024,0.905,379.5
128,True,1024,0.681,252.2
128,False,2048,1.459,471.1
128,True,2048,1.025,335.2
128,False,4096,2.608,527.0
128,True,4096,1.717,400.3
128,False,8192,4.904,560.5
128,True,8192,3.127,439.5
128,False,16384,9.756,563.5
128,True,16384,5.932,463.4
"""

# 使用pandas的read_csv和StringIO来读取字符串数据
df_base = pd.read_csv(io.StringIO(baseline_data))
df_desc = pd.read_csv(io.StringIO(descending_data))
df_pd = pd.read_csv(io.StringIO(persistent_descending_data))
df_shift = pd.read_csv(io.StringIO(shift_data))
df_shift_persistent = pd.read_csv(io.StringIO(shift_data_persistent))


# ==================== 2. 准备数据 ====================
# 为每个DataFrame添加一个'Method'列
df_base['Method'] = 'baseline'
df_desc['Method'] = 'descending'
df_pd['Method'] = 'persistent+descending'
df_shift['Method'] = 'shift'
df_shift_persistent['Method'] = 'shift+persistent'

# 将所有数据合并到一个DataFrame中
all_data = pd.concat([df_base, df_desc, df_pd, df_shift, df_shift_persistent], ignore_index=True)

# 创建一个用于x轴标签的组合列
all_data['Configuration'] = 'd' + all_data['headdim'].astype(str) + '-s' + all_data['seqlen'].astype(str)

# 按照您的要求，定义柱状图的顺序
method_order = ['baseline', 'descending', 'persistent+descending', 'shift', 'shift+persistent']
all_data['Method'] = pd.Categorical(all_data['Method'], categories=method_order, ordered=True)

# 为了确保图表x轴的顺序正确，我们根据headdim和seqlen进行排序
all_data.sort_values(by=['headdim', 'seqlen', 'Method'], inplace=True)


# ==================== 3. 绘图 ====================
# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体，以防标题乱码
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 创建一个包含两个子图的图表 (2行, 1列)
fig, axes = plt.subplots(2, 1, figsize=(20, 16))

# 设置一个总标题
fig.suptitle('Fav3 Backward Pass Performance Comparison (TFLOPS)', fontsize=22)

# ----- 绘制 Causal = False 的子图 -----
data_false = all_data[all_data['causal'] == False]
sns.barplot(
    ax=axes[0],
    x='Configuration',
    y='Fav3 bwd (TFLOPS)',
    hue='Method',
    data=data_false,
    hue_order=method_order
)
axes[0].set_title('Causal = False', fontsize=18)
axes[0].set_xlabel('', fontsize=14)  # 顶部图表不需要x轴标签
axes[0].set_ylabel('Backward Pass (TFLOPS)', fontsize=14)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend(title='Method', fontsize=12)

# ----- 绘制 Causal = True 的子图 -----
data_true = all_data[all_data['causal'] == True]
sns.barplot(
    ax=axes[1],
    x='Configuration',
    y='Fav3 bwd (TFLOPS)',
    hue='Method',
    data=data_true,
    hue_order=method_order
)
axes[1].set_title('Causal = True', fontsize=18)
axes[1].set_xlabel('Configuration (Head Dimension - Sequence Length)', fontsize=14)
axes[1].set_ylabel('Backward Pass (TFLOPS)', fontsize=14)
axes[1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].legend(title='Method', fontsize=12)


# 调整布局以防止标签重叠并显示图表
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 为总标题留出空间
plt.savefig('fav3_backward_pass_comparison.png')
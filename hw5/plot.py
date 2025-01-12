import matplotlib.pyplot as plt
import numpy as np

# 数据
matrix_sizes = ["256x256x256", "512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096"]
cpu_times = [49929, 393642, 3532150, 34694060, 283390609]  # 最后一个值缺失
gemm_times = [167, 205, 249, 251, 426]
wmma_times = [16, 14, 19, 17, 19]

# 处理缺失值
cpu_times = [time / 1000000 for time in cpu_times]

# 绘制柱状图
bar_width = 0.25
index = np.arange(len(matrix_sizes))

plt.figure(figsize=(12, 6))
plt.bar(index, cpu_times, bar_width, label='CPU (s)')
plt.bar(index + bar_width, gemm_times, bar_width, label='GEMM (us)')
plt.bar(index + 2 * bar_width, wmma_times, bar_width, label='WMMA (us)')

plt.xlabel('Matrix Size')
plt.ylabel('Time for CPU is (s), Time for GPU is (us)')
plt.title('Runtime Comparison of CPU, GEMM, and WMMA')
plt.xticks(index + bar_width, matrix_sizes, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

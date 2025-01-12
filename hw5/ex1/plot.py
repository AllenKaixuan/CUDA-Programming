import matplotlib.pyplot as plt

sizes = []
cpu_times = []
gemm_times = []
tiled_8x16_times = []
tiled_16x16_times = []
tiled_32x16_times = []

with open('result.log', 'r') as f:
    for line in f:
        if "Running test for matrix size" in line:
            size = line.split(":")[1].strip()
            sizes.append(size)
        elif "Timing - CPU." in line:
            cpu_times.append(float(line.split("Elasped")[1].strip().split()[0]) / 1000000) 
        elif "Timing - CUDA gemm." in line:
            gemm_times.append(float(line.split("Elasped")[1].strip().split()[0]) )  
        elif "CUDA kernel (Tiled) with tile size 8x16" in line:
            next_line = next(f)
            tiled_8x16_times.append(float(next_line.split("Elasped")[1].strip().split()[0]) )  
        elif "CUDA kernel (Tiled) with tile size 16x16" in line:
            next_line = next(f)
            tiled_16x16_times.append(float(next_line.split("Elasped")[1].strip().split()[0]) )  
        elif "CUDA kernel (Tiled) with tile size 32x16" in line:
            next_line = next(f)
            tiled_32x16_times.append(float(next_line.split("Elasped")[1].strip().split()[0]) )  

#assert len(sizes) == len(cpu_times) == len(gemm_times) == len(tiled_8x16_times) == len(tiled_16x16_times) == len(tiled_32x16_times), "Data length mismatch"
print(sizes)
print(cpu_times)
print(gemm_times)
print(tiled_8x16_times)
print(tiled_16x16_times)
print(tiled_32x16_times)

plt.figure(figsize=(12, 8))
bar_width = 0.15
index = range(len(sizes))

plt.bar(index, cpu_times, bar_width, label='CPU(s)')
plt.bar([i + bar_width for i in index], gemm_times, bar_width, label='GEMM(us)')
plt.bar([i + 2 * bar_width for i in index], tiled_8x16_times, bar_width, label='Tiled GEMM 8x16(us)')
plt.bar([i + 3 * bar_width for i in index], tiled_16x16_times, bar_width, label='Tiled GEMM 16x16(us)')
plt.bar([i + 4 * bar_width for i in index], tiled_32x16_times, bar_width, label='Tiled GEMM 32x16(us)')

plt.xlabel('Matrix Size')
plt.ylabel('Time (us) for GPU, Time (s) for CPU')
plt.title('Performance Comparison')
plt.xticks([i + 2 * bar_width for i in index], sizes, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
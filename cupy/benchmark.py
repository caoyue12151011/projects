import time
import cupy as cp
import numpy as np
import timeit
import matplotlib.pyplot as plt

# 设置不同的数组大小进行测试
array_sizes = [100, 500, 1000]

# 准备存储时间的列表
cupy_times = []
numpy_times = []

# 对每个数组大小进行基准测试
for n in array_sizes:
    print(f"Testing array size: {n}")

    # 预热GPU和CPU
    cupy_array = cp.random.rand(n, n)
    numpy_array = np.random.rand(n, n)
    cp.matmul(cupy_array, cupy_array)
    np.matmul(numpy_array, numpy_array)

    # 基准测试CuPy和NumPy的矩阵乘法操作
    cupy_time = timeit.timeit(
        "result = cp.matmul(cupy_array, cupy_array)", globals=globals(), number=10
    )
    numpy_time = timeit.timeit(
        "result = np.matmul(numpy_array, numpy_array)", globals=globals(), number=10
    )

    # 收集结果
    cupy_times.append(cupy_time)
    numpy_times.append(numpy_time)
    print(f"Array size: {n}, CuPy time: {cupy_time}, NumPy time: {numpy_time}")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, cupy_times, label="CuPy", marker="o", color="blue")
plt.plot(array_sizes, numpy_times, label="NumPy", marker="x", color="orange")
plt.xlabel("Array Size")
plt.ylabel("Average Time (seconds)")
plt.title("CuPy vs NumPy Performance Benchmark")
plt.legend()
plt.grid(True)

# 保存图表为PNG文件
plt.savefig("performance_benchmark.png", dpi=300, bbox_inches="tight")

# 关闭图形，以释放资源
plt.close()

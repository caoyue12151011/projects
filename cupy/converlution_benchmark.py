import numpy as np
import cupy as cp
import scipy.ndimage as sp_ndimage
import cupyx.scipy.ndimage as cp_ndimage
import time
import timeit
import matplotlib.pyplot as plt


def warmup_convolve_2d():
    np_array = np.random.rand(100, 100)
    np_filter = np.random.rand(10, 10)
    cp_array = cp.asarray(np_array)
    cp_filter = cp.asarray(np_filter)

    sp_ndimage.convolve(np_array, np_filter, mode="mirror")
    cp_ndimage.convolve(cp_array, cp_filter, mode="mirror")


def benchmark_convolve_2d(size, iterations=10):
    setup = f"""
import numpy as np
import cupy as cp
import scipy.ndimage
import cupyx.scipy.ndimage as cp_ndimage

size = {size}
iterations = {iterations}

np_array = np.random.rand(size, size)
np_filter = np.random.rand(size // 10, size // 10)
cp_array = cp.asarray(np_array)
cp_filter = cp.asarray(np_filter)
    """

    # NumPy卷积测试
    np_statement = "scipy.ndimage.convolve(np_array, np_filter, mode='mirror')"
    np_time = timeit.timeit(np_statement, setup=setup, number=iterations) / iterations

    # CuPy卷积测试
    cp_statement = "cp_ndimage.convolve(cp_array, cp_filter, mode='mirror')"
    cp_time = timeit.timeit(cp_statement, setup=setup, number=iterations) / iterations

    # 计算时间比率
    ratio = np_time / cp_time if cp_time else float("inf")

    return np_time, cp_time, ratio


warmup_convolve_2d()

sizes = [100, 320, 640]
results = []

print(
    f"{'Size':<10}{'NumPy Time (s)':<20}{'CuPy Time (s)':<20}{'NumPy/CuPy Ratio':<20}"
)
for size in sizes:
    np_time, cp_time, ratio = benchmark_convolve_2d(size)
    results.append((size, np_time, cp_time, ratio))
    print(f"{size:<10}{np_time:<20.6f}{cp_time:<20.6f}{ratio:<20.2f}")

results = list(zip(*results))
sizes, np_times, cp_times, ratios = results

plt.plot(sizes, np_times, label="NumPy Time")
plt.plot(sizes, cp_times, label="CuPy Time")
plt.plot(
    sizes,
    [np_times[i] / cp_times[i] for i in range(len(sizes))],
    label="Ratio (NumPy/CuPy)",
    linestyle="--",
    marker="o",
)
plt.xlabel("Array Size")
plt.ylabel("Time (s)")
plt.title("2D Convolution Speed: NumPy vs CuPy")
plt.legend()
plt.savefig("performance_benchmark.png", dpi=300, bbox_inches="tight")
plt.close()

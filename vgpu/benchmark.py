import torch
import vgpu
from torch.profiler import profile, schedule, record_function, ProfilerActivity
import time
import math
import numpy as np
from tabulate import tabulate


ITERS = 1000
NUM_SAMPLES = 100
M = int(8)
KS = [int(2**i) for i in range(3, 16, 4)]  # 8 a 32k
N = int(128)
THREADS = 1

OPERATORS = [
    torch.mm,
    torch.vgpu.ops.mmult_mkl,
]


def benchmark_operator(operator, mat1, mat2, n_iterations, log=False):
    operator_name = f"{operator.__module__}.{operator.__name__} ({mat1.device})"
    # Warm up
    for _ in range(n_iterations):
        operator(mat1, mat2)
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        operator(mat1, mat2)
    end_time = time.perf_counter()
    average_time = (end_time - start_time) / n_iterations
    average_time_us = 1e6 * average_time
    if log:
        print(
            f"[{operator_name}] Average time per matrix multiplication (µs): { average_time_us }"
        )
    return average_time_us


if __name__ == "__main__":
    # Run benchmark
    torch.set_num_threads(THREADS)
    print(f"Number of threads: {torch.get_num_threads()}")

    for k in KS:
        print(f"\nMultiplication size: [{M} x {k}] x [{k} x {N}]")

        # Create two random matrices in float32
        mat1_cpu = torch.randn(M, k)
        mat2_cpu = torch.randn(k, N)

        mat1_vgpu = torch.randn(M, k, device="vgpu")
        mat2_vgpu = torch.randn(k, N, device="vgpu")

        results_header = ["Operator", "Min", "Median", "Mean"]
        results = []
        for operator in OPERATORS:
            operator_name = f"{operator.__module__}.{operator.__name__}"
            results_cpu = []
            results_vgpu = []

            for _ in range(NUM_SAMPLES):
                average_computation_time_cpu = benchmark_operator(
                    operator, mat1_cpu, mat2_cpu, ITERS
                )
                results_cpu.append(average_computation_time_cpu)
                average_computation_time_vgpu = benchmark_operator(
                    operator, mat1_vgpu, mat2_vgpu, ITERS
                )
                results_vgpu.append(average_computation_time_vgpu)

            results.append(
                [
                    f"{operator_name} (cpu)",
                    np.min(results_cpu),
                    np.median(results_cpu),
                    np.mean(results_cpu),
                ]
            )
            results.append(
                [
                    f"{operator_name} (vgpu)",
                    np.min(results_vgpu),
                    np.median(results_vgpu),
                    np.mean(results_vgpu),
                ]
            )
        print(
            tabulate(
                results,
                headers=results_header,
                tablefmt="rounded_outline",
                floatfmt=".2f",
                numalign="right",
                stralign="center",
            )
        )

        # for operator in operators:
        #     for i in range(N):
        #         operator(a, b)  # Warm up
        #     with profile(
        #         activities=[ProfilerActivity.CPU],
        #     ) as prof:
        #         for i in range(N):
        #             with record_function("operator_call"):
        #                 operator(a, b)
        #
        #     print(prof.key_averages())

import os
import numpy as np
import tvm
from tvm import te, auto_scheduler
import time

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(n, m, l, dtype):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A', dtype = dtype)
    B = te.placeholder((l, m), name='B', dtype = dtype)
    C = te.compute((n, m),
                   lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                   name='C')
    return [A, B, C]


target = tvm.target.Target("llvm -mcpu=core-avx2")

start = time.time()
for i in range(5, 12):
    size = 2 ** i
    print(f"size: {size}")
    N = L = M = size
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(N, L, M, "float32"), target=target)

    log_file = "matmul_" + str(size) + ".json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
end = time.time()

print("elapsed time:")
print(end - start)
import numpy as np
import tvm
import timeit
from tvm import te
from tvm import te, auto_scheduler


target = tvm.target.Target(target="llvm -mcpu=core-avx2", host="llvm -mcpu=core-avx2")
dev = tvm.device(target.kind.name, 0)

def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape."""
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def bench_workload(workload):
    """Benchmark a workload

    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

def bench_matmul_tvm(func, sizes):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev, number=nrepeats)
        return timer(a, b, c).mean * nrepeats

    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        a, b, c = get_abc((n, n), lambda x: tvm.nd.array(x, dev))
        times.append(bench_workload(workload))
    return 2 * sizes ** 3 / 1e9 / np.array(times)

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(n, m, l, dtype):
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A', dtype = dtype)
    B = te.placeholder((l, m), name='B', dtype = dtype)
    C = te.compute((n, m),
                   lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                   name='C')
    return [A, B, C]

# n = 1024
# task = tvm.auto_scheduler.SearchTask(func=matmul, args=(n, n, n, "float32"), target=target)
# print(task.print_best("log_files/2000_trials/matmul_1024.json"))

def optimized_matmul(n):
    A, B, C = matmul(n, n, n, "float32")
    s = te.create_schedule(C.op)

    if n == 32:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_local, = s.cache_write([C], "local")
        C_local_x_c, C_local_y_c, C_local_k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
        C_local_x_c_o_i, C_local_x_c_i = s[C_local].split(C_local_x_c, factor=2)
        C_local_x_c_o_o_i, C_local_x_c_o_i = s[C_local].split(C_local_x_c_o_i, factor=2)
        C_local_x_c_o_o_o, C_local_x_c_o_o_i = s[C_local].split(C_local_x_c_o_o_i, factor=8)
        C_local_y_c_o_i, C_local_y_c_i = s[C_local].split(C_local_y_c, factor=16)
        C_local_y_c_o_o_i, C_local_y_c_o_i = s[C_local].split(C_local_y_c_o_i, factor=1)
        C_local_y_c_o_o_o, C_local_y_c_o_o_i = s[C_local].split(C_local_y_c_o_o_i, factor=2)
        C_local_k_o, C_local_k_i = s[C_local].split(C_local_k, factor=2)
        s[C_local].reorder(C_local_x_c_o_o_o, C_local_y_c_o_o_o, C_local_x_c_o_o_i, C_local_y_c_o_o_i, C_local_k_o, C_local_x_c_o_i, C_local_y_c_o_i, C_local_k_i, C_local_x_c_i, C_local_y_c_i)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=4)
        C_x_o_o, C_x_o_i = s[C].split(C_x_o_i, factor=8)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=16)
        C_y_o_o, C_y_o_i = s[C].split(C_y_o_i, factor=2)
        s[C].reorder(C_x_o_o, C_y_o_o, C_x_o_i, C_y_o_i, C_x_i, C_y_i)
        s[C_local].compute_at(s[C], C_y_o_i)
        C_x_o_o = s[C].fuse(C_x_o_o)
        s[C].parallel(C_x_o_o)
        s[C_local].pragma(C_local_x_c_o_o_o, "auto_unroll_max_step", 0)
        s[C_local].pragma(C_local_x_c_o_o_o, "unroll_explicit", True)
        s[C_local].vectorize(C_local_y_c_i)
        s[C].vectorize(C_y_i)
    elif n == 64:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=1)
        C_x_o_o_i, C_x_o_i = s[C].split(C_x_o_i, factor=1)
        C_x_o_o_o, C_x_o_o_i = s[C].split(C_x_o_o_i, factor=2)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=64)
        C_y_o_o_i, C_y_o_i = s[C].split(C_y_o_i, factor=1)
        C_y_o_o_o, C_y_o_o_i = s[C].split(C_y_o_o_i, factor=1)
        C_k_o, C_k_i = s[C].split(C_k, factor=64)
        s[C].reorder(C_x_o_o_o, C_y_o_o_o, C_x_o_o_i, C_y_o_o_i, C_k_o, C_x_o_i, C_y_o_i, C_k_i, C_x_i, C_y_i)
        C_x_o_o_o_y_o_o_o_fused_x_o_o_i_fused_y_o_o_i_fused = s[C].fuse(C_x_o_o_o, C_y_o_o_o, C_x_o_o_i, C_y_o_o_i)
        s[C].parallel(C_x_o_o_o_y_o_o_o_fused_x_o_o_i_fused_y_o_o_i_fused)
        s[C].pragma(C_x_o_o_o_y_o_o_o_fused_x_o_o_i_fused_y_o_o_i_fused, "auto_unroll_max_step", 0)
        s[C].pragma(C_x_o_o_o_y_o_o_o_fused_x_o_o_i_fused_y_o_o_i_fused, "unroll_explicit", True)
    elif n == 128:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=4)
        C_x_o_o_i, C_x_o_i = s[C].split(C_x_o_i, factor=1)
        C_x_o_o_o, C_x_o_o_i = s[C].split(C_x_o_o_i, factor=1)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=16)
        C_y_o_o_i, C_y_o_i = s[C].split(C_y_o_i, factor=1)
        C_y_o_o_o, C_y_o_o_i = s[C].split(C_y_o_o_i, factor=1)
        C_k_o, C_k_i = s[C].split(C_k, factor=16)
        s[C].reorder(C_x_o_o_o, C_y_o_o_o, C_x_o_o_i, C_y_o_o_i, C_k_o, C_x_o_i, C_y_o_i, C_k_i, C_x_i, C_y_i)
        C_x_o_o_o_y_o_o_o_fused = s[C].fuse(C_x_o_o_o, C_y_o_o_o)
        s[C].parallel(C_x_o_o_o_y_o_o_o_fused)
        s[C].pragma(C_x_o_o_o_y_o_o_o_fused, "auto_unroll_max_step", 16)
        s[C].pragma(C_x_o_o_o_y_o_o_o_fused, "unroll_explicit", True)
        s[C].vectorize(C_y_i)
    elif n == 256:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=1)
        C_x_o_o_i, C_x_o_i = s[C].split(C_x_o_i, factor=4)
        C_x_o_o_o, C_x_o_o_i = s[C].split(C_x_o_o_i, factor=1)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=16)
        C_y_o_o_i, C_y_o_i = s[C].split(C_y_o_i, factor=1)
        C_y_o_o_o, C_y_o_o_i = s[C].split(C_y_o_o_i, factor=16)
        C_k_o, C_k_i = s[C].split(C_k, factor=16)
        s[C].reorder(C_x_o_o_o, C_y_o_o_o, C_x_o_o_i, C_y_o_o_i, C_k_o, C_x_o_i, C_y_o_i, C_k_i, C_x_i, C_y_i)
        C_x_o_o_o = s[C].fuse(C_x_o_o_o)
        s[C].parallel(C_x_o_o_o)
        s[C].pragma(C_x_o_o_o, "auto_unroll_max_step", 64)
        s[C].pragma(C_x_o_o_o, "unroll_explicit", True)
        s[C].vectorize(C_y_i)
    elif n == 512:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_local, = s.cache_write([C], "local")
        C_local_x_c, C_local_y_c, C_local_k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
        C_local_x_c_o_i, C_local_x_c_i = s[C_local].split(C_local_x_c, factor=2)
        C_local_x_c_o_o_i, C_local_x_c_o_i = s[C_local].split(C_local_x_c_o_i, factor=64)
        C_local_x_c_o_o_o, C_local_x_c_o_o_i = s[C_local].split(C_local_x_c_o_o_i, factor=4)
        C_local_y_c_o_i, C_local_y_c_i = s[C_local].split(C_local_y_c, factor=32)
        C_local_y_c_o_o_i, C_local_y_c_o_i = s[C_local].split(C_local_y_c_o_i, factor=1)
        C_local_y_c_o_o_o, C_local_y_c_o_o_i = s[C_local].split(C_local_y_c_o_o_i, factor=1)
        C_local_k_o, C_local_k_i = s[C_local].split(C_local_k, factor=16)
        s[C_local].reorder(C_local_x_c_o_o_o, C_local_y_c_o_o_o, C_local_x_c_o_o_i, C_local_y_c_o_o_i, C_local_k_o, C_local_x_c_o_i, C_local_y_c_o_i, C_local_k_i, C_local_x_c_i, C_local_y_c_i)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=128)
        C_x_o_o, C_x_o_i = s[C].split(C_x_o_i, factor=4)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=32)
        C_y_o_o, C_y_o_i = s[C].split(C_y_o_i, factor=1)
        s[C].reorder(C_x_o_o, C_y_o_o, C_x_o_i, C_y_o_i, C_x_i, C_y_i)
        s[C_local].compute_at(s[C], C_y_o_i)
        C_x_o_o_y_o_o_fused_x_o_i_fused_y_o_i_fused = s[C].fuse(C_x_o_o, C_y_o_o, C_x_o_i, C_y_o_i)
        s[C].parallel(C_x_o_o_y_o_o_fused_x_o_i_fused_y_o_i_fused)
        s[C_local].pragma(C_local_x_c_o_o_o, "auto_unroll_max_step", 64)
        s[C_local].pragma(C_local_x_c_o_o_o, "unroll_explicit", True)
        s[C_local].vectorize(C_local_y_c_i)
        s[C].vectorize(C_y_i)
    elif n == 1024:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_local, = s.cache_write([C], "local")
        C_local_x_c, C_local_y_c, C_local_k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
        C_local_x_c_o_i, C_local_x_c_i = s[C_local].split(C_local_x_c, factor=4)
        C_local_x_c_o_o_i, C_local_x_c_o_i = s[C_local].split(C_local_x_c_o_i, factor=4)
        C_local_x_c_o_o_o, C_local_x_c_o_o_i = s[C_local].split(C_local_x_c_o_o_i, factor=1)
        C_local_y_c_o_i, C_local_y_c_i = s[C_local].split(C_local_y_c, factor=16)
        C_local_y_c_o_o_i, C_local_y_c_o_i = s[C_local].split(C_local_y_c_o_i, factor=64)
        C_local_y_c_o_o_o, C_local_y_c_o_o_i = s[C_local].split(C_local_y_c_o_o_i, factor=1)
        C_local_k_o, C_local_k_i = s[C_local].split(C_local_k, factor=8)
        s[C_local].reorder(C_local_x_c_o_o_o, C_local_y_c_o_o_o, C_local_x_c_o_o_i, C_local_y_c_o_o_i, C_local_k_o, C_local_x_c_o_i, C_local_y_c_o_i, C_local_k_i, C_local_x_c_i, C_local_y_c_i)
        C_x_o, C_x_i = s[C].split(C_x, factor=16)
        C_y_o, C_y_i = s[C].split(C_y, factor=1024)
        s[C].reorder(C_x_o, C_y_o, C_x_i, C_y_i)
        s[C_local].compute_at(s[C], C_y_o)
        C_x_o = s[C].fuse(C_x_o)
        s[C].parallel(C_x_o)
        s[C_local].pragma(C_local_x_c_o_o_o, "auto_unroll_max_step", 64)
        s[C_local].pragma(C_local_x_c_o_o_o, "unroll_explicit", True)
        s[C_local].vectorize(C_local_y_c_i)
    elif n == 2048:
        C_x, C_y, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
        C_local, = s.cache_write([C], "local")
        C_local_x_c, C_local_y_c, C_local_k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
        C_local_x_c_o_i, C_local_x_c_i = s[C_local].split(C_local_x_c, factor=4)
        C_local_x_c_o_o_i, C_local_x_c_o_i = s[C_local].split(C_local_x_c_o_i, factor=16)
        C_local_x_c_o_o_o, C_local_x_c_o_o_i = s[C_local].split(C_local_x_c_o_o_i, factor=2)
        C_local_y_c_o_i, C_local_y_c_i = s[C_local].split(C_local_y_c, factor=16)
        C_local_y_c_o_o_i, C_local_y_c_o_i = s[C_local].split(C_local_y_c_o_i, factor=64)
        C_local_y_c_o_o_o, C_local_y_c_o_o_i = s[C_local].split(C_local_y_c_o_o_i, factor=2)
        C_local_k_o, C_local_k_i = s[C_local].split(C_local_k, factor=16)
        s[C_local].reorder(C_local_x_c_o_o_o, C_local_y_c_o_o_o, C_local_x_c_o_o_i, C_local_y_c_o_o_i, C_local_k_o, C_local_x_c_o_i, C_local_y_c_o_i, C_local_k_i, C_local_x_c_i, C_local_y_c_i)
        C_x_o_i, C_x_i = s[C].split(C_x, factor=64)
        C_x_o_o, C_x_o_i = s[C].split(C_x_o_i, factor=2)
        C_y_o_i, C_y_i = s[C].split(C_y, factor=1024)
        C_y_o_o, C_y_o_i = s[C].split(C_y_o_i, factor=2)
        s[C].reorder(C_x_o_o, C_y_o_o, C_x_o_i, C_y_o_i, C_x_i, C_y_i)
        s[C_local].compute_at(s[C], C_y_o_i)
        C_x_o_o_y_o_o_fused_x_o_i_fused_y_o_i_fused = s[C].fuse(C_x_o_o, C_y_o_o, C_x_o_i, C_y_o_i)
        s[C].parallel(C_x_o_o_y_o_o_fused_x_o_i_fused_y_o_i_fused)
        s[C_local].pragma(C_local_x_c_o_o_o, "auto_unroll_max_step", 64)
        s[C_local].pragma(C_local_x_c_o_o_o, "unroll_explicit", True)
        s[C_local].vectorize(C_local_y_c_i)
        s[C].vectorize(C_y_i)
    else:
        print(f"Invalid input size: {n}")
        exit()
    return s, (A, B, C)

sizes = 2 ** np.arange(5, 12, 1)
auto_scheduling_gflops = bench_matmul_tvm(optimized_matmul, sizes)
print(auto_scheduling_gflops)
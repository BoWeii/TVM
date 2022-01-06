import numpy as np
import tvm
import timeit
from tvm import te
from matplotlib import pyplot as plt
from IPython import display
import d2ltvm

tgt = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(tgt.kind.name, 0)


def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(4.5, 3)):
    """Plot multiple lines"""

    display.set_matplotlib_formats('svg')
    # InlineBackend.figure_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    X, Y = np.array(X), np.array(Y)
    if X.shape != Y.shape: X = [X] * len(Y)
    if not fmts: fmts = ['-'] * len(X)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()
    # plt.plot()
    plt.show()


# Save to the d2ltvm package
def plot_gflops(sizes, gflops, legend, xlabel='Size'):
    plot(sizes, gflops, xlabel=xlabel, ylabel='GFLOPS',
         xscale='log', yscale='log',
         legend=legend, fmts=['--'] * (len(gflops) - 1) + ['-'])


# Save to the d2ltvm package.
def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                   lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                   name='C')
    return A, B, C


# Save to the d2ltvm package.
def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape."""
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c


# Save to the d2ltvm package.
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


# Save to the d2ltvm package.
def np_matmul_timer(n):
    timer = timeit.Timer(setup='import numpy as np\n'
                               'import d2ltvm\n'
                               'a, b, c = d2ltvm.get_abc(%s)' % str((n, n)),
                         stmt='np.dot(a, b, out=c)')
    return timer.timeit


sizes = 2 ** np.arange(5, 12, 1)
exe_times = [bench_workload(np_matmul_timer(n)) for n in sizes]
np_gflops = 2 * sizes ** 3 / 1e9 / np.array(exe_times)


# Save to the d2ltvm package.
def bench_matmul_tvm(func, sizes):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev, number=nrepeats)
        return timer(a, b, c).mean * nrepeats

    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], tgt)
        a, b, c = get_abc((n, n), lambda x: tvm.nd.array(x, dev))
        times.append(bench_workload(workload))
    return 2 * sizes ** 3 / 1e9 / np.array(times)


def default(n):
    A, B, C = matmul(n, n, n)
    return te.create_schedule(C.op), (A, B, C)


# default part
s, args = default(64)
mod = tvm.build(s, args)
print(tvm.lower(s, args, simple_mode=True))
default_gflops = bench_matmul_tvm(default, sizes)


def reorder(n):
    s, (A, B, C) = default(n)
    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    return s, (A, B, C)


# reorder part
s, args = reorder(64)
mod = tvm.build(s, args)
print(tvm.lower(s, args, simple_mode=True))
reorder_gflops = bench_matmul_tvm(reorder, sizes)


def parallel(n):
    s, (A, B, C) = reorder(n)
    s[C].parallel(C.op.axis[0])
    return s, (A, B, C)


# parallel part
s, args = parallel(64)
print(tvm.lower(s, args, simple_mode=True))
parallel_gflops = bench_matmul_tvm(parallel, sizes)


def block(n):
    tx, ty, tk = 32, 32, 4
    s, (A, B, C) = parallel(n)
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # Optimize the computation of each block
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk)
    s[C].reorder(ko, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].unroll(ki)
    return s, (A, B, C)


# block part
s, args = block(64)
print(tvm.lower(s, args, simple_mode=True))
blocked_gflops = bench_matmul_tvm(block, sizes)


def cached_block(n):
    tx, ty, tk = 32, 32, 4
    s, (A, B, C) = parallel(n)
    CachedC = s.cache_write(C, 'local')
    # Same as before, first tile by blocks, and then parallelize the
    # computation of each block
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # Use the write cache for the output of the xy axis, namely a block.
    s[CachedC].compute_at(s[C], xy)
    # Same as before to optimize the computation of a block .
    xc, yc = s[CachedC].op.axis
    ko, ki = s[CachedC].split(CachedC.op.reduce_axis[0], factor=tk)
    s[CachedC].reorder(ko, xc, ki, yc)
    s[CachedC].unroll(ki)
    s[CachedC].vectorize(yc)
    return s, (A, B, C)


# cached_block part
s, args = cached_block(64)
print(tvm.lower(s, args, simple_mode=True))
cached_gflops = bench_matmul_tvm(cached_block, sizes)

# draw the result
plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops, parallel_gflops, blocked_gflops, cached_gflops],
            ['numpy', 'default', 'reorder', 'parallel', 'block', 'cache'])

print("end!!!!")

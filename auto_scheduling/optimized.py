def matmul(n, m, l, dtype):
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A', dtype = dtype)
    B = te.placeholder((l, m), name='B', dtype = dtype)
    C = te.compute((n, m),
                   lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                   name='C')
    return [A, B, C]
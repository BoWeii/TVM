import numpy as np
import tvm
from tvm import te

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

def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape."""
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

n = 100
A, B, C = matmul(n, n, n)
s = te.create_schedule(C.op)
# print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B, C])

a, b, c = get_abc((100, 100), tvm.nd.array)
mod(a, b, c)
np.testing.assert_allclose(np.dot(a.asnumpy(), b.asnumpy()),
                           c.asnumpy(), atol=1e-5)
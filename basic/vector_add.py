import numpy as np
import tvm
from tvm import te # te stands for tensor expression

def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape."""
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def vector_add(n):
    """TVM expression for vector add"""
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

n = 100
A, B, C = vector_add(n)

s = te.create_schedule(C.op)
mod = tvm.build(s, [A, B, C])

# A compiled a module can be saved into disk,
mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)

# and then loaded back later.
loaded_mod = tvm.runtime.load_module(mod_fname)

# Verify the results.
a, b, c = get_abc(100, tvm.nd.array)
loaded_mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, language_level=3

cimport numpy as cnp
import numpy as np

cdef extern from "lapjv_seeded.h":
    int c_lapjv_seeded "lapjv_seeded"(
        const double* C, int n_rows, int n_cols,
        long long* x, long long* y,
        const double* u_seed, const double* v_seed,
        double eps)

def lapjv_seeded(cnp.ndarray[cnp.double_t, ndim=2, mode='c'] C,
                 cnp.ndarray[cnp.double_t, ndim=1, mode='c'] u,
                 cnp.ndarray[cnp.double_t, ndim=1, mode='c'] v,
                 double eps=1e-12):
    cdef int n = C.shape[0]
    cdef int m = C.shape[1]
    if u.shape[0] != n or v.shape[0] != m:
        raise ValueError("u/v sizes must match C")
    cdef cnp.ndarray[cnp.int64_t, ndim=1] x = np.full((n,), -1, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] y = np.full((m,), -1, dtype=np.int64)
    cdef int ret = c_lapjv_seeded(&C[0,0], n, m, <long long*>&x[0], <long long*>&y[0],
                                  &u[0], &v[0], eps)
    if ret != 0:
        if ret == -3:
            raise ValueError("Infeasible seed potentials: C - u - v has negatives")
        raise RuntimeError(f"lapjv_seeded internal error (code {ret})")
    cost = float(np.sum(C[np.arange(n), x]))
    return x, y, cost
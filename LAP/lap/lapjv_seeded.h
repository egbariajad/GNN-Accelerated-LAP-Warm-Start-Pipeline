// lap/lapjv_seeded.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int lapjv_seeded(
    const double* C, int n_rows, int n_cols,
    long long* x, long long* y,
    const double* u_seed, const double* v_seed,
    double eps
);

#ifdef __cplusplus
}
#endif

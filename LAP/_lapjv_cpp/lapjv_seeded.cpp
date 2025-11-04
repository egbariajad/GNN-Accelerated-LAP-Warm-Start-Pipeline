#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include "lapjv.h"
#include "lapjv_seeded.h"

static inline bool feasible(const double* C, int n, int m,
                            const double* u, const double* v, double eps) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (C[i * m + j] - u[i] - v[j] < -eps) return false;
        }
    }
    return true;
}

extern "C" int lapjv_seeded(
    const double* C, int n_rows, int n_cols,
    long long* x, long long* y,
    const double* u_seed, const double* v_seed,
    double eps)
{
    if (n_rows <= 0 || n_cols <= 0) return -2;
    // Only support square dense for now (consistent with JV internal API)
    if (n_rows != n_cols) return -4;
    const int n = n_rows;

    // Initialize matching arrays and copy seeded potentials first
    std::vector<int> x_i(n, -1);
    std::vector<int> y_i(n, -1);
    std::vector<double> u(u_seed, u_seed + n);  // Copy u_seed
    std::vector<double> v(v_seed, v_seed + n);  // Copy v_seed

    // OPTIMIZATION 4: Feasibility projection - ensure u[i] + v[j] <= C[i,j] for all i,j
    // Project dual potentials to feasible region before any other checks
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double violation = u[i] + v[j] - C[i * n + j];
            if (violation > eps) {
                // Share the violation equally between u[i] and v[j]
                double adjustment = violation / 2.0;
                u[i] -= adjustment;
                v[j] -= adjustment;
            }
        }
    }

    // Verify feasibility after projection
    if (!feasible(C, n_rows, n_cols, u.data(), v.data(), eps)) {
        return -3; // infeasible potentials even after projection
    }

    // Prepare cost matrix for JV internal format
    std::vector<double*> cost(n);
    for (int i = 0; i < n; ++i) {
        cost[i] = const_cast<double*>(&C[i * n]);
    }

    // Initialize matching arrays and copy seeded potentials
    // (already done above before feasibility projection)

    // OPTIMIZATION 1: Row tightening - force one zero per row
    // u[i] = min_j(C[i,j] - v[j]) ensures min_j(C[i,j] - u[i] - v[j]) = 0
    for (int i = 0; i < n; ++i) {
        double min_reduced = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j) {
            double reduced_cost = C[i * n + j] - v[j];
            min_reduced = std::min(min_reduced, reduced_cost);
        }
        u[i] = min_reduced;
    }

    // OPTIMIZATION 3: Use looser tightness threshold for numerical stability
    const double tight_eps = std::max(eps, 1e-9);

    // Greedy equality matching phase - match zero reduced-cost edges
    std::vector<bool> col_used(n, false);
    int matched_count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (col_used[j]) continue;
            double reduced_cost = C[i * n + j] - u[i] - v[j];
            if (std::abs(reduced_cost) <= tight_eps) {
                x_i[i] = j;
                y_i[j] = i;
                col_used[j] = true;
                matched_count++;
                break;
            }
        }
    }

    // Collect unmatched rows and columns
    std::vector<int> free_rows, free_cols;
    for (int i = 0; i < n; ++i) {
        if (x_i[i] < 0) free_rows.push_back(i);
    }
    for (int j = 0; j < n; ++j) {
        if (y_i[j] < 0) free_cols.push_back(j);
    }

    // OPTIMIZATION 5: Fallback trigger - if zero-edge density too low, use regular JV
    int total_tight_edges = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double reduced_cost = C[i * n + j] - u[i] - v[j];
            if (std::abs(reduced_cost) <= tight_eps) {
                total_tight_edges++;
            }
        }
    }

    // If tight edge density < 1.2n, fall back to unseeded JV
    if (total_tight_edges < 1.2 * n) {
        extern int lapjv_internal(const unsigned int n, double* cost[], int* x, int* y);
        std::vector<int> temp_x(n), temp_y(n);
        int ret = lapjv_internal((unsigned int)n, cost.data(), temp_x.data(), temp_y.data());
        if (ret != 0) return ret;
        
        for (int i = 0; i < n; ++i) x[i] = temp_x[i];
        for (int j = 0; j < n; ++j) y[j] = temp_y[j];
        return 0;
    }

    // If everything is matched, we're done
    if (free_rows.empty()) {
        for (int i = 0; i < n; ++i) x[i] = x_i[i];
        for (int j = 0; j < n; ++j) y[j] = y_i[j];
        return 0;
    }

    // OPTIMIZATION 2: Micro-ARR on unmatched rows - create second zeros
    // This is a simplified version of augmenting row reduction for unmatched rows only
    for (int free_i : free_rows) {
        // Find two minimum reduced costs for this row
        double min1 = std::numeric_limits<double>::infinity();
        double min2 = std::numeric_limits<double>::infinity();
        int j1 = -1;
        
        for (int j = 0; j < n; ++j) {
            double reduced_cost = C[free_i * n + j] - u[free_i] - v[j];
            if (reduced_cost < min1) {
                min2 = min1;
                min1 = reduced_cost;
                j1 = j;
            } else if (reduced_cost < min2) {
                min2 = reduced_cost;
            }
        }
        
        // If we can create a second zero by adjusting v[j1], do it
        if (j1 >= 0 && min2 - min1 > tight_eps && std::find(free_cols.begin(), free_cols.end(), j1) != free_cols.end()) {
            double delta = min2 - min1;
            v[j1] += delta;
            // This creates a second zero in row free_i at column j1
        }
    }

    // For unmatched rows, call ONLY the augmenting path phase (_ca_dense)
    extern int _ca_dense(const unsigned int n, double* cost[], const unsigned int n_free_rows,
                         int* free_rows, int* x, int* y, double* v);

    int ret = _ca_dense((unsigned int)n, cost.data(), (unsigned int)free_rows.size(),
                        free_rows.data(), x_i.data(), y_i.data(), v.data());
    if (ret != 0) return ret;

    // Copy final result to output arrays
    for (int i = 0; i < n; ++i) x[i] = x_i[i];
    for (int j = 0; j < n; ++j) y[j] = y_i[j];
    return 0;
}

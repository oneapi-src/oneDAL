/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <limits>

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"

namespace oneapi::dal::backend::primitives {

template <bool beta, typename Float, ndorder xlayout, ndorder ylayout>
inline sycl::event beta_copy_transform(sycl::queue& queue,
                                       const ndview<Float, 2, xlayout>& src,
                                       ndview<Float, 2, ylayout>& dst,
                                       const event_vector& dependencies) {
    ONEDAL_ASSERT(src.has_data());
    const auto shape = dst.get_shape();
    ONEDAL_ASSERT(dst.has_mutable_data());

    ONEDAL_ASSERT(shape.at(0) == src.get_dimension(0));
    ONEDAL_ASSERT(shape.at(1) == src.get_dimension(1) + !beta);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);

        auto src_ndx = make_ndindexer(src);
        auto dst_ndx = make_ndindexer(dst);

        const auto w = shape.at(1);
        const auto range = shape.to_range();
        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto r = idx[0];
            const auto c = idx[1];

            if (c == 0) {
                if constexpr (beta) {
                    dst_ndx.at(r, c) = src_ndx.at(r, w - 1);
                }
                else {
                    dst_ndx.at(r, c) = Float(0);
                }
            }
            else {
                dst_ndx.at(r, c) = src_ndx.at(r, c - 1);
            }
        });
    });
}

/*
This is an adaptation of the CPU version, which can be found in this file:
cpp/daal/src/algorithms/service_kernel_math.h

It solves a linear system A*x = b
where 'b' might be a matrix (multiple RHS)

It is intended as a fallback for solving linear regression, where these
matrices are obtained as follows:
    A = t(X)*X
    b = t(X)*y

It can handle systems that are not positive semi-definite, so it's used
as a fallback when Cholesky fails or when it involves too small numbers
(which tends to happen when floating point error results in a slightly
positive matrix when it should have zero determinant in theory).
*/
template <mkl::uplo uplo, typename Float>
sycl::event solve_spectral_decomposition(
    sycl::queue& queue,
    ndview<Float, 2>& A, // t(X)*X, LHS, gets overwritten
    sycl::event& event_A,
    ndview<Float, 2>& b, // t(X)*y, RHS, solution is outputted here
    sycl::event& event_b,
    const std::int64_t dim_A,
    const std::int64_t nrhs) {
    constexpr auto alloc = sycl::usm::alloc::device;

    /* Decompose: A = Q * diag(l) * t(Q) */
    /* Note: for NRHS>1, this will overallocate in order to reuse the memory as buffer later on */
    auto eigenvalues = array<Float>::empty(queue, dim_A * nrhs, alloc);
    auto eigenvalues_view = ndview<Float, 1>::wrap(eigenvalues);
    sycl::event syevd_event =
        syevd<mkl::job::vec, uplo, Float>(queue, dim_A, A, dim_A, eigenvalues_view, { event_A });
    const Float eps = std::numeric_limits<Float>::epsilon();

    /* Discard too small singular values */
    std::int64_t num_discarded;
    {
        /* This is placed inside a block because the array created here is not used any further */
        auto eigenvalues_cpu = eigenvalues_view.to_host(queue, { syevd_event });
        const Float* eigenvalues_cpu_ptr = eigenvalues_cpu.get_data();
        const Float largest_ev = eigenvalues_cpu_ptr[dim_A - 1];
        if (largest_ev <= eps) {
            throw std::runtime_error(
                "Could not solve linear system. Problem has too small singular values.");
        }
        const Float component_threshold = eps * largest_ev;
        for (num_discarded = 0; num_discarded < dim_A - 1; num_discarded++) {
            if (eigenvalues_cpu_ptr[num_discarded] > component_threshold)
                break;
        }
    }

    /* Create the square root of the inverse: Qis = Q * diag(1 / sqrt(l)) */
    std::int64_t num_taken = dim_A - num_discarded;
    auto ev_mutable = eigenvalues.get_mutable_data();
    sycl::event inv_sqrt_eigenvalues_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(syevd_event);
        h.parallel_for(num_taken, [=](const auto& i) {
            const std::size_t ix = i + num_discarded;
            ev_mutable[ix] = sycl::sqrt(Float(1) / ev_mutable[ix]);
        });
    });

    auto Q_mutable = A.get_mutable_data();
    sycl::event inv_sqrt_eigenvectors_event = queue.submit([&](sycl::handler& h) {
        const auto range = oneapi::dal::backend::make_range_2d(num_taken, dim_A);
        h.depends_on(inv_sqrt_eigenvalues_event);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t col = id[0] + num_discarded;
            const std::size_t row = id[1];
            Q_mutable[row + col * dim_A] *= ev_mutable[col];
        });
    });

    /* Now calculate the actual solution: Qis * Qis' * B */
    const std::int64_t eigenvectors_offset = num_discarded * dim_A;
    if (nrhs == 1) {
        sycl::event gemv_right_event =
            mkl::blas::column_major::gemv(queue,
                                          mkl::transpose::trans,
                                          dim_A,
                                          num_taken,
                                          Float(1),
                                          Q_mutable + eigenvectors_offset,
                                          dim_A,
                                          b.get_data(),
                                          1,
                                          Float(0),
                                          ev_mutable,
                                          1,
                                          { inv_sqrt_eigenvectors_event, event_b });
        return mkl::blas::column_major::gemv(queue,
                                             mkl::transpose::nontrans,
                                             dim_A,
                                             num_taken,
                                             Float(1),
                                             Q_mutable + eigenvectors_offset,
                                             dim_A,
                                             ev_mutable,
                                             1,
                                             Float(0),
                                             b.get_mutable_data(),
                                             1,
                                             { gemv_right_event });
    }

    else {
        sycl::event gemm_right_event =
            mkl::blas::column_major::gemm(queue,
                                          mkl::transpose::trans,
                                          mkl::transpose::nontrans,
                                          num_taken,
                                          nrhs,
                                          dim_A,
                                          Float(1),
                                          Q_mutable + eigenvectors_offset,
                                          dim_A,
                                          b.get_data(),
                                          dim_A,
                                          Float(0),
                                          ev_mutable,
                                          num_taken,
                                          { inv_sqrt_eigenvectors_event, event_b });
        return mkl::blas::column_major::gemm(queue,
                                             mkl::transpose::nontrans,
                                             mkl::transpose::nontrans,
                                             dim_A,
                                             nrhs,
                                             num_taken,
                                             Float(1),
                                             Q_mutable + eigenvectors_offset,
                                             dim_A,
                                             ev_mutable,
                                             num_taken,
                                             Float(0),
                                             b.get_mutable_data(),
                                             dim_A,
                                             { gemm_right_event });
    }
}

/* Returns the minimum value among entries in the diagonal of a square matrix */
template <typename Float>
Float diagonal_minimum(sycl::queue& queue,
                       const Float* Matrix,
                       const std::int64_t dim_matrix,
                       sycl::event& event_Matrix) {
    constexpr auto alloc = sycl::usm::alloc::device;
    auto diag_min_holder = array<Float>::empty(queue, 1, alloc);
    sycl::event diag_min_holder_init = queue.submit([&](sycl::handler& h) {
        Float* diag_min_ptr = diag_min_holder.get_mutable_data();
        h.parallel_for(1, [=](const auto& i) {
            diag_min_ptr[i] = std::numeric_limits<Float>::infinity();
        });
    });
    sycl::event diag_min_event = queue.submit([&](sycl::handler& h) {
        auto min_reduction = sycl::reduction(diag_min_holder.get_mutable_data(), sycl::minimum<>());
        h.depends_on({ diag_min_holder_init, event_Matrix });
        h.parallel_for(dim_matrix, min_reduction, [=](const auto& i, auto& min_obj) {
            min_obj.combine(Matrix[i * (dim_matrix + 1)]);
        });
    });
    return ndview<Float, 1>::wrap(diag_min_holder).at_device(queue, 0, { diag_min_event });
}

template <mkl::uplo uplo, bool beta, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event solve_system(sycl::queue& queue,
                         const ndview<Float, 2, xlayout>& xtx,
                         const ndview<Float, 2, ylayout>& xty,
                         ndview<Float, 2, ndorder::c>& final_xtx,
                         ndview<Float, 2, ndorder::c>& final_xty,
                         const event_vector& dependencies) {
    constexpr auto alloc = sycl::usm::alloc::device;

    auto [nxty, xty_event] = copy<ndorder::c, Float, ylayout, alloc>(queue, xty, dependencies);
    auto [nxtx, xtx_event] = copy<ndorder::c, Float, xlayout, alloc>(queue, xtx, dependencies);
    const std::int64_t dim_xtx = xtx.get_dimension(0);

    sycl::event solution_event;

    opt_array<Float> dummy{};
    try {
        sycl::event potrf_event = potrf_factorization<uplo>(queue, nxtx, dummy, { xtx_event });
        const Float diag_min = diagonal_minimum(queue, nxtx.get_data(), dim_xtx, potrf_event);
        if (diag_min <= 1e-6)
            throw mkl::lapack::computation_error("", "", 0);
        solution_event = potrs_solution<uplo>(queue, nxtx, nxty, dummy, { potrf_event, xty_event });
    }
    catch (mkl::lapack::computation_error& ex) {
        const std::int64_t nrhs = nxty.get_dimension(0);
        /* Note: this templated version of 'copy' reuses the layout that was specified in the previous copy */
        sycl::event xtx_event_new = copy(queue, nxtx, xtx, dependencies);
        sycl::event xty_event_new = copy(queue, nxty, xty, dependencies);

        solution_event = solve_spectral_decomposition<uplo, Float>(queue,
                                                                   nxtx,
                                                                   xtx_event_new,
                                                                   nxty,
                                                                   xty_event_new,
                                                                   dim_xtx,
                                                                   nrhs);
    }

    return beta_copy_transform<beta>(queue, nxty, final_xty, { solution_event });
}

#define INSTANTIATE(U, B, F, XL, YL)                                 \
    template sycl::event solve_system<U, B>(sycl::queue&,            \
                                            const ndview<F, 2, XL>&, \
                                            const ndview<F, 2, YL>&, \
                                            ndview<F, 2>&,           \
                                            ndview<F, 2>&,           \
                                            const event_vector&);

#define INSTANTIATE_YL(U, B, F, XL)      \
    INSTANTIATE(U, B, F, XL, ndorder::f) \
    INSTANTIATE(U, B, F, XL, ndorder::c)

#define INSTANTIATE_XL(U, B, F)         \
    INSTANTIATE_YL(U, B, F, ndorder::f) \
    INSTANTIATE_YL(U, B, F, ndorder::c)

#define INSTANTIATE_F(U, B)     \
    INSTANTIATE_XL(U, B, float) \
    INSTANTIATE_XL(U, B, double)

#define INSTANTIATE_B(U)   \
    INSTANTIATE_F(U, true) \
    INSTANTIATE_F(U, false)

INSTANTIATE_B(mkl::uplo::upper)
INSTANTIATE_B(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives

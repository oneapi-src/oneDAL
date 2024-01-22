/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"

namespace oneapi::dal::pca::backend {

#ifdef ONEDAL_DATA_PARALLEL

using alloc = sycl::usm::alloc;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

// Common

/// Compute sums wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The input data
template <typename Float>
auto compute_sums(sycl::queue& queue,
                  const pr::ndview<Float, 2>& data,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, queue);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(data.get_dimension(1) > 0);

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);
    constexpr pr::sum<Float> binary{};
    constexpr pr::identity<Float> unary{};
    auto sums_event = pr::reduce_by_columns(queue, data, sums, binary, unary, deps);
    return std::make_tuple(sums, sums_event);
}

/// Compute means wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  sums  The precomputed sums
/// @param[in]  row_count  The number of rows
template <typename Float>
auto compute_means(sycl::queue& q,
                   const pr::ndview<Float, 1>& sums,
                   std::int64_t row_count,
                   const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(sums.get_dimension(0) > 0);

    const std::int64_t column_count = sums.get_dimension(0);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto means_event = pr::means(q, row_count, sums, means, deps);
    return std::make_tuple(means, means_event);
}

/// Compute explained variances wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  eigenvalues  The eigenvalues
/// @param[in]  vars  The variances
template <typename Float>
auto compute_explained_variances_on_host(sycl::queue& q,
                                         pr::ndarray<Float, 1> eigenvalues,
                                         pr::ndarray<Float, 1> vars,
                                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_explained_variances_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);
    const std::int64_t column_count = vars.get_dimension(0);
    auto explained_variances_ratio = pr::ndarray<Float, 1>::empty(component_count);

    auto eigvals_ptr = eigenvalues.get_data();
    auto vars_ptr = vars.get_data();
    auto explained_variances_ratio_ptr = explained_variances_ratio.get_mutable_data();

    Float sum = 0;
    for (std::int64_t i = 0; i < column_count; ++i) {
        sum += vars_ptr[i];
    }
    ONEDAL_ASSERT(sum > 0);
    const Float inverse_sum = 1.0 / sum;
    for (std::int64_t i = 0; i < component_count; ++i) {
        explained_variances_ratio_ptr[i] = eigvals_ptr[i] * inverse_sum;
    }
    return explained_variances_ratio;
}

// Cov and Precomputed methods

/// Compute covariance matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  xtx  The result of xtx of the input data
/// @param[in]  sums  The sums
/// @param[in]  bias  The bias value
template <typename Float>
auto compute_covariance(sycl::queue& q,
                        std::int64_t row_count,
                        const pr::ndview<Float, 2>& xtx,
                        const pr::ndarray<Float, 1>& sums,
                        bool bias,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_covariance, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());
    ONEDAL_ASSERT(xtx.get_dimension(1) > 0);

    const std::int64_t column_count = xtx.get_dimension(1);

    auto cov = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, cov, xtx, { deps });

    auto cov_event = pr::covariance(q, row_count, sums, cov, bias, { copy_event });
    return std::make_tuple(cov, cov_event);
}

/// Compute correaltion matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The numer of rows
/// @param[in]  xtx  The result of xtx of the input data
/// @param[in]  sums  The sums
template <typename Float>
auto compute_correlation(sycl::queue& q,
                         std::int64_t row_count,
                         const pr::ndview<Float, 2>& xtx,
                         const pr::ndarray<Float, 1>& sums,
                         const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());
    ONEDAL_ASSERT(xtx.get_dimension(1) > 0);

    const std::int64_t column_count = xtx.get_dimension(1);

    auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, corr, xtx, { deps });

    auto corr_event = pr::correlation(q, row_count, sums, corr, { copy_event });

    return std::make_tuple(corr, corr_event);
}

/// Compute crossproduct wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The input data
template <typename Float>
auto compute_crossproduct(sycl::queue& q,
                          const pr::ndview<Float, 2>& data,
                          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_crossproduct, q);
    ONEDAL_ASSERT(data.has_data());

    const std::int64_t column_count = data.get_dimension(1);
    auto xtx = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q);
        gemm_event = gemm(q, data.t(), data, xtx, Float(1.0), Float(0.0));
        gemm_event.wait_and_throw();
    }

    return std::make_tuple(xtx, gemm_event);
}

/// Compute variances from covariance matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  cov  The covariance matrix
template <typename Float>
auto compute_variances(sycl::queue& q,
                       const pr::ndview<Float, 2>& cov,
                       const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_vars, q);
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(cov.get_dimension(0) > 0);
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");

    auto column_count = cov.get_dimension(0);
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto vars_event = pr::variances(q, cov, vars, deps);
    return std::make_tuple(vars, vars_event);
}

/// Compute correaltion from covariance matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  cov  The covariance matrix
template <typename Float>
auto compute_correlation_from_covariance(sycl::queue& q,
                                         std::int64_t row_count,
                                         const pr::ndview<Float, 2>& cov,
                                         const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, q);
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(cov.get_dimension(0) > 0);
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");

    const std::int64_t column_count = cov.get_dimension(1);

    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);

    auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    const bool bias = false; // Currently we use only unbiased covariance for PCA computation.
    auto corr_event = pr::correlation_from_covariance(q, row_count, cov, corr, tmp, bias, deps);

    return std::make_tuple(corr, corr_event);
}

// SVD method
/// Compute eigenvectors and eigenvalues wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  corr  The correlation matrix
/// @param[in]  component_count  The number of descriptor components
template <typename Float>
auto compute_eigenvectors_on_host(sycl::queue& q,
                                  pr::ndarray<Float, 2>&& corr,
                                  std::int64_t component_count,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvectors_on_host);
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(corr.get_dimension(0) > 0);
    const std::int64_t column_count = corr.get_dimension(0);

    auto eigvecs = pr::ndarray<Float, 2>::empty({ component_count, column_count });
    auto eigvals = pr::ndarray<Float, 1>::empty(component_count);
    auto host_corr = corr.to_host(q, deps);
    pr::sym_eigvals_descending(host_corr, component_count, eigvecs, eigvals);

    return std::make_tuple(eigvecs, eigvals);
}

/// Compute eigenvalues from singular values wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  singular_values  The singluar values
/// @param[in]  row_count  The number of rows
template <typename Float>
auto compute_eigenvalues_on_device(sycl::queue& q,
                                   pr::ndarray<Float, 1> singular_values,
                                   std::int64_t row_count,
                                   const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvalues_on_device);

    const std::int64_t component_count = singular_values.get_dimension(0);

    auto eigenvalues = pr::ndarray<Float, 1>::empty(component_count);

    auto singular_values_ptr = singular_values.get_data();
    auto eigvals_ptr = eigenvalues.get_mutable_data();

    const Float factor = row_count - 1;
    for (std::int64_t i = 0; i < component_count; ++i) {
        eigvals_ptr[i] = singular_values_ptr[i] * singular_values_ptr[i] / factor;
    }
    return eigenvalues;
}

/// Compute singular values from eigen values wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  eigenvalues  The eigen values
/// @param[in]  row_count  The number of rows
template <typename Float>
auto compute_singular_values_on_host(sycl::queue& q,
                                     pr::ndarray<Float, 1> eigenvalues,
                                     std::int64_t row_count,
                                     const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_singular_values_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);

    auto singular_values = pr::ndarray<Float, 1>::empty(component_count);

    auto eigvals_ptr = eigenvalues.get_data();
    auto singular_values_ptr = singular_values.get_mutable_data();

    const Float factor = row_count - 1;
    for (std::int64_t i = 0; i < component_count; ++i) {
        singular_values_ptr[i] = std::sqrt(factor * eigvals_ptr[i]);
    }
    return singular_values;
}

/// Compute  U/VT sliced data wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The data necessary to be sliced
/// @param[in]  component_count  The number of descriptor components
/// @param[in]  column_count  The number of columns
template <typename Float>
auto slice_data(sycl::queue& q,
                const pr::ndview<Float, 2>& data,
                std::int64_t component_count,
                std::int64_t column_count,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    const std::int64_t column_count_local = data.get_dimension(1);
    auto data_to_compute =
        pr::ndarray<Float, 2>::empty(q, { component_count, column_count }, alloc::device);
    auto data_to_compute_ptr = data_to_compute.get_mutable_data();
    auto data_ptr = data.get_data();
    auto event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(component_count, column_count);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t row = id[0];
            const std::int64_t column = id[1];
            data_to_compute_ptr[row * column_count + column] =
                data_ptr[row * column_count_local + column];
        });
    });
    return std::make_tuple(data_to_compute, event);
}

/// Compute  mean centered data wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The input data
/// @param[in]  means  The precomputed means
template <typename Float>
auto get_centered(sycl::queue& q,
                  pr::ndview<Float, 2>& data,
                  const pr::ndview<Float, 1>& means,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_centered_data, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto centered_data_ptr = data.get_mutable_data();
    auto means_ptr = means.get_data();

    auto centered_event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t i = id[0];
            const std::size_t j = id[1];
            centered_data_ptr[i * column_count + j] =
                centered_data_ptr[i * column_count + j] - means_ptr[j];
        });
    });
    return centered_event;
}

/// Compute scaled data wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The mean centered data
/// @param[in]  means  The variances
template <typename Float>
auto get_scaled(sycl::queue& q,
                pr::ndview<Float, 2>& data,
                const pr::ndview<Float, 1>& variances,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_scaled_data, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto scaled_data_ptr = data.get_mutable_data();
    auto variances_ptr = variances.get_data();

    auto scaled_event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t i = id[0];
            const std::size_t j = id[1];
            const Float sqrt_var = sycl::sqrt(variances_ptr[j]);
            const Float inv_var =
                sqrt_var < std::numeric_limits<Float>::epsilon() ? 0 : 1 / sqrt_var;
            scaled_data_ptr[i * column_count + j] = scaled_data_ptr[i * column_count + j] * inv_var;
        });
    });
    return scaled_event;
}

/// Compute variances on device wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The mean centered data
template <typename Float>
auto compute_variances_device(sycl::queue& q,
                              const pr::ndview<Float, 2>& data,
                              const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);

    constexpr pr::sum<Float> binary;
    constexpr pr::square<Float> unary;

    auto vars_event_ = pr::reduce_by_columns(q, data, vars, binary, unary, deps);
    auto vars_ptr = vars.get_mutable_data();
    auto vars_event = q.submit([&](sycl::handler& h) {
        h.depends_on(vars_event_);
        const auto range = bk::make_range_1d(column_count);
        h.parallel_for(range, [=](sycl::id<1> id) {
            const std::int64_t i = id[0];

            vars_ptr[i] = vars_ptr[i] / (row_count - 1);
        });
    });
    return std::make_tuple(vars, vars_event);
}

// Online Cov method

/// Compute init step of online algorithm wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
template <typename Float>
auto init(sycl::queue& q,
          const std::int64_t row_count,
          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(init_partial_results, q);

    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);

    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto init_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<1>(1);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_nobs_ptr[0] = row_count;
        });
    });

    return std::make_tuple(result_nobs, init_event);
}

/// Update partial results of  online algorithm wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  crossproducts  The crossproduct of current iteration
/// @param[in]  sums  The sums of current iteration
/// @param[in]  current_crossproducts  The crossproduct of previous iterations
/// @param[in]  current_sums  The sums of previous iterations
/// @param[in]  current_nobs  The number of observations of previous iterations
/// @param[in]  row_count  The number of rows
template <typename Float>
auto update_partial_results(sycl::queue& q,
                            const pr::ndview<Float, 2>& crossproducts,
                            const pr::ndview<Float, 1>& sums,
                            const pr::ndview<Float, 2>& current_crossproducts,
                            const pr::ndview<Float, 1>& current_sums,
                            const pr::ndview<Float, 1>& current_nobs,
                            const std::int64_t row_count,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_results, q);

    auto column_count = crossproducts.get_dimension(1);
    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_crossproducts =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_crossproducts_ptr = result_crossproducts.get_mutable_data();
    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto current_crossproducts_data = current_crossproducts.get_data();
    auto current_sums_data = current_sums.get_data();
    auto current_nobs_data = current_nobs.get_data();

    auto crossproducts_data = crossproducts.get_data();
    auto sums_data = sums.get_data();

    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<2>(column_count, column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t row = id[0], col = id[1];

            result_crossproducts_ptr[row * column_count + col] =
                crossproducts_data[row * column_count + col] +
                current_crossproducts_data[row * column_count + col];

            if (static_cast<std::int64_t>(col) < column_count) {
                result_sums_ptr[col] = sums_data[col] + current_sums_data[col];
            }

            if (row == 0 && col == 0) {
                result_nobs_ptr[0] = row_count + current_nobs_data[0];
            }
        });
    });

    return std::make_tuple(result_sums, result_crossproducts, result_nobs, update_event);
}

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL

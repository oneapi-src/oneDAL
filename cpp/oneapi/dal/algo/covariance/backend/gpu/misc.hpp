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

namespace oneapi::dal::covariance::backend {

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
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, q);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(data.get_dimension(1) > 0);

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    constexpr pr::sum<Float> binary{};
    constexpr pr::identity<Float> unary{};
    auto sums_event = pr::reduce_by_columns(q, data, sums, binary, unary, deps);
    return std::make_tuple(sums, sums_event);
}

/// Compute means wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  sums  The means
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

/// Compute covariance matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  xtx  The xtx
/// @param[in]  sums  The sums
/// @param[in]  bias  The bias values
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

/// Compute correlation matrix wrapper
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  xtx  The xtx
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

/// Compute crossproduct matrix wrapper
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

// Online

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

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL

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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel_cov_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using model_t = model<task::dim_reduction>;
using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

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

    auto corr_event = pr::correlation_from_covariance(q, row_count, cov, corr, tmp, deps);

    return std::make_tuple(corr, corr_event);
}

template <typename Float>
auto compute_eigenvectors_on_host(sycl::queue& q,
                                  pr::ndarray<Float, 2>&& corr,
                                  std::int64_t component_count,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvectors_on_host);
    ONEDAL_ASSERT(corr.has_mutable_data());
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

template <typename Float>
result_t train_kernel_cov_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();
    std::int64_t row_count = data.get_row_count();
    auto rows_count_global = row_count;
    ONEDAL_ASSERT(data.get_column_count() > 0);
    std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);
    auto sums = pr::ndarray<Float, 1>::empty(q_, { column_count }, alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q_, data_nd, sums, pr::sum<Float>{}, pr::identity<Float>{}, {});

    {
        ONEDAL_PROFILER_TASK(allreduce_sums, q_);
        comm_.allreduce(sums.flatten(q_, { sums_event }), spmd::reduce_op::sum).wait();
    }
    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0));
        gemm_event.wait_and_throw();
    }

    {
        ONEDAL_PROFILER_TASK(allreduce_xtx, q_);
        comm_.allreduce(xtx.flatten(q_, { gemm_event }), spmd::reduce_op::sum).wait();
    }
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global);
        comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
    }

    if (desc.get_result_options().test(result_options::means)) {
        auto means = pr::ndarray<Float, 1>::empty(q_, { column_count }, alloc::device);
        auto means_event = pr::means(q, rows_count_global, sums, means, gemm_event);
        result.set_means(homogen_table::wrap(means.flatten(q_), 1, column_count));
    }

    auto [cov, cov_event] =
        pr::compute_covariance(q_, rows_count_global, xtx, sums, { gemm_event });
    if (desc.get_result_options().test(result_options::vars)) {
        auto [vars, vars_event] = compute_variances(q_, cov, { cov_event });
        vars_event.wait_and_throw();
        result.set_variances(homogen_table::wrap(vars.flatten(q_), 1, column_count));
    }
    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [corr, corr_event] =
            compute_correlation_from_covariance(q_, rows_count_global, cov, { gemm_event });

        auto [eigvecs, eigvals] =
            compute_eigenvectors_on_host(q_, std::move(corr), component_count, { corr_event });
        if (desc.get_result_options().test(result_options::eigenvalues)) {
            result.set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count));
        }

        if (desc.get_deterministic()) {
            sign_flip(eigvecs);
        }
        if (desc.get_result_options().test(result_options::eigenvectors)) {
            const auto model = model_t{}.set_eigenvectors(
                homogen_table::wrap(eigvecs.flatten(), component_count, column_count));
            result.set_model(model);
        }
    }

    return result;
}

template class train_kernel_cov_impl<float>;
template class train_kernel_cov_impl<double>;

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL

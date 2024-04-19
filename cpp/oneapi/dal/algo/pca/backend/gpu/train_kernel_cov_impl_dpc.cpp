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
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;

using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
result_t train_kernel_cov_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.

    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();

    ONEDAL_ASSERT(data.get_row_count() > 0);
    std::int64_t row_count = data.get_row_count();
    auto rows_count_global = row_count;
    ONEDAL_ASSERT(row_count > 0);

    ONEDAL_ASSERT(data.get_column_count() > 0);
    std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);

    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());
    dal::detail::check_mul_overflow(column_count, component_count);

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto [sums, sums_event] = compute_sums(q_, data_nd);

    {
        ONEDAL_PROFILER_TASK(allreduce_sums, q_);
        comm_.allreduce(sums.flatten(q_, { sums_event }), spmd::reduce_op::sum).wait();
    }

    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0), { sums_event });
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

    sycl::event means_event;
    if (desc.get_result_options().test(result_options::means)) {
        auto [means, means_event] = compute_means(q_, sums, rows_count_global, { gemm_event });
        result.set_means(homogen_table::wrap(means.flatten(q_, { means_event }), 1, column_count));
    }

    auto [cov, cov_event] =
        compute_covariance(q_, rows_count_global, xtx, sums, bias, { gemm_event });

    auto [vars, vars_event] = compute_variances(q_, cov, { cov_event, means_event });

    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(
            homogen_table::wrap(vars.flatten(q_, { vars_event }), 1, column_count));
    }

    auto data_to_compute = cov;

    sycl::event corr_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        auto corr = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
        corr_event =
            pr::correlation_from_covariance(q_, rows_count_global, cov, corr, bias, { cov_event });
        data_to_compute = corr;
    }

    auto [eigvecs, eigvals] = compute_eigenvectors_on_host(q_,
                                                           std::move(data_to_compute),
                                                           component_count,
                                                           { cov_event, corr_event, vars_event });

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::singular_values)) {
        auto singular_values =
            compute_singular_values_on_host(q_,
                                            eigvals,
                                            rows_count_global,
                                            { cov_event, corr_event, vars_event });
        result.set_singular_values(
            homogen_table::wrap(singular_values.flatten(), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        auto vars_host = vars.to_host(q_);
        auto explained_variances_ratio =
            compute_explained_variances_on_host(q_,
                                                eigvals,
                                                vars_host,
                                                { cov_event, corr_event, vars_event });
        result.set_explained_variances_ratio(
            homogen_table::wrap(explained_variances_ratio.flatten(), 1, component_count));
    }

    if (desc.get_deterministic()) {
        sign_flip(eigvecs);
    }

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        result.set_eigenvectors(
            homogen_table::wrap(eigvecs.flatten(), component_count, column_count));
    }

    return result;
}

template class train_kernel_cov_impl<float>;
template class train_kernel_cov_impl<double>;

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL

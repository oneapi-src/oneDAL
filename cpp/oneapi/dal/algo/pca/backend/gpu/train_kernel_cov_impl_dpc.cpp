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
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, q);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(0 < data.get_dimension(1));

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
    return std::make_tuple(sums, reduce_event);
}

template <typename Float>
auto compute_means(sycl::queue& q,
                   std::int64_t row_count,
                   const pr::ndview<Float, 1>& sums,
                   const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(sums.get_dimension(0) > 0);

    const std::int64_t column_count = sums.get_dimension(0);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto means_event = pr::means(q, row_count, sums, means, deps);
    return std::make_tuple(means, means_event);
}

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
auto compute_covariance(sycl::queue& q,
                        std::int64_t row_count,
                        const pr::ndview<Float, 2>& xtx,
                        const pr::ndarray<Float, 1>& sums,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_covariance, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());
    ONEDAL_ASSERT(xtx.get_dimension(1) > 0);

    const std::int64_t column_count = xtx.get_dimension(1);

    auto cov = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, cov, xtx, { deps });

    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.
    auto cov_event = pr::covariance(q, row_count, sums, cov, bias, { copy_event });
    return std::make_tuple(cov, cov_event);
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

    const bool bias = false; // Currently we use only unbiased covariance for PCA computation.
    auto corr_event = pr::correlation_from_covariance(q, row_count, cov, corr, tmp, bias, deps);

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

    if (desc.get_result_options().test(result_options::means)) {
        auto [means, means_event] = compute_means(q_, rows_count_global, sums, { gemm_event });
        result.set_means(homogen_table::wrap(means.flatten(q_, { means_event }), 1, column_count));
    }

    auto [cov, cov_event] = compute_covariance(q_, rows_count_global, xtx, sums, { gemm_event });

    auto [vars, vars_event] = compute_variances(q_, cov, { cov_event });
    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(
            homogen_table::wrap(vars.flatten(q_, { vars_event }), 1, column_count));
    }
    auto data_to_compute = cov;
    sycl::event corr_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        pr::ndarray<Float, 2> corr{};
        std::tie(corr, corr_event) =
            compute_correlation_from_covariance(q_, rows_count_global, cov, { cov_event });
        corr_event.wait_and_throw();
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
                                            row_count,
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

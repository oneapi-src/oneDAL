/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/gpu/finalize_train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using model_t = model<task::dim_reduction>;
using task_t = task::dim_reduction;
using input_t = partial_train_result<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

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
    ONEDAL_ASSERT(0 < sum);
    const Float inverse_sum = 1.0 / sum;
    for (std::int64_t i = 0; i < component_count; ++i) {
        explained_variances_ratio_ptr[i] = eigvals_ptr[i] * inverse_sum;
    }
    return explained_variances_ratio;
}
template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_t& desc,
                                const partial_train_result<Task>& input) {
    auto& q = ctx.get_queue();
    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.
    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();
    const std::int64_t component_count =
        get_component_count(desc, input.get_partial_crossproduct());

    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto nobs_host = pr::table2ndarray<Float>(q, input.get_partial_n_rows());
    auto rows_count_global = nobs_host.get_data()[0];

    const auto sums =
        pr::table2ndarray_1d<Float>(q, input.get_partial_sum(), sycl::usm::alloc::device);
    sycl::event means_event;
    if (desc.get_result_options().test(result_options::means)) {
        ONEDAL_PROFILER_TASK(compute_means, q);
        auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
        means_event = pr::means(q, rows_count_global, sums, means, {});
        result.set_means(homogen_table::wrap(means.flatten(q, { means_event }), 1, column_count));
    }

    const auto xtx =
        pr::table2ndarray<Float>(q, input.get_partial_crossproduct(), sycl::usm::alloc::device);
    auto cov = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
    sycl::event copy_event;
    {
        ONEDAL_PROFILER_TASK(copy_xtx, q);
        copy_event = copy(q, cov, xtx, { means_event });
    }

    sycl::event cov_event;
    {
        ONEDAL_PROFILER_TASK(compute_covariance_matrix, q);
        cov_event = pr::covariance(q, rows_count_global, sums, cov, bias, { copy_event });
    }

    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    sycl::event vars_event;
    {
        ONEDAL_PROFILER_TASK(compute_means, q);
        vars_event = pr::variances(q, cov, vars, { cov_event, means_event });
    }
    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(homogen_table::wrap(vars.flatten(q, { vars_event }), 1, column_count));
    }
    auto data_to_compute = cov;

    sycl::event corr_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

        corr_event =
            pr::correlation_from_covariance(q, rows_count_global, cov, corr, bias, { cov_event });

        data_to_compute = corr;
    }

    auto [eigvecs, eigvals] = compute_eigenvectors_on_host(q,
                                                           std::move(data_to_compute),
                                                           component_count,
                                                           { corr_event, vars_event, cov_event });
    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::singular_values)) {
        auto singular_values =
            compute_singular_values_on_host(q,
                                            eigvals,
                                            rows_count_global,
                                            { corr_event, vars_event, cov_event });
        result.set_singular_values(
            homogen_table::wrap(singular_values.flatten(), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        auto vars_host = vars.to_host(q);
        auto explained_variances_ratio =
            compute_explained_variances_on_host(q,
                                                eigvals,
                                                vars_host,
                                                { corr_event, vars_event, cov_event });
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

template <typename Float>
struct finalize_train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float, task::dim_reduction>(ctx, desc, input);
    }
};

template struct finalize_train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct finalize_train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

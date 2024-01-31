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
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"
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

using task_t = task::dim_reduction;
using input_t = partial_train_result<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_t& desc,
                                const partial_train_result<Task>& input) {
    auto& q = ctx.get_queue();

    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.

    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count =
        get_component_count(desc, input.get_partial_crossproduct());
    ONEDAL_ASSERT(component_count > 0);

    dal::detail::check_mul_overflow(column_count, column_count);

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto nobs_host = pr::table2ndarray<Float>(q, input.get_partial_n_rows());
    auto rows_count_global = nobs_host.get_data()[0];

    const auto sums =
        pr::table2ndarray_1d<Float>(q, input.get_partial_sum(), sycl::usm::alloc::device);

    if (desc.get_result_options().test(result_options::means)) {
        auto [means, means_event] = compute_means(q, sums, rows_count_global, {});
        result.set_means(homogen_table::wrap(means.flatten(q, { means_event }), 1, column_count));
    }

    const auto xtx =
        pr::table2ndarray<Float>(q, input.get_partial_crossproduct(), sycl::usm::alloc::device);
    auto [cov, cov_event] = compute_covariance(q, rows_count_global, xtx, sums, {});

    auto [vars, vars_event] = compute_variances(q, cov, { cov_event });

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
struct finalize_train_kernel_gpu<Float, method::cov, task_t> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float, task_t>(ctx, desc, input);
    }
};

template struct finalize_train_kernel_gpu<float, method::cov, task_t>;
template struct finalize_train_kernel_gpu<double, method::cov, task_t>;

} // namespace oneapi::dal::pca::backend

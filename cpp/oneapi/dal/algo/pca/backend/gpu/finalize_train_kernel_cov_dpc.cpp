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

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/sign_flip.hpp"
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
    eigvecs = eigvecs.to_device(q, deps);
    return std::make_tuple(eigvecs, eigvals);
}

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

    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);

    auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, corr, xtx, { deps });

    auto corr_event = pr::correlation(q, row_count, sums, corr, tmp, { copy_event });

    auto smart_event = bk::smart_event{ corr_event }.attach(tmp);
    return std::make_tuple(corr, smart_event);
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_t& desc,
                                const detail::train_parameters<Task>& params,
                                const partial_train_result<Task>& input) {
    auto& q = ctx.get_queue();

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
    const auto xtx =
        pr::table2ndarray<Float>(q, input.get_partial_crossproduct(), sycl::usm::alloc::device);

    auto [corr, corr_event] = compute_correlation(q, rows_count_global, xtx, sums);

    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [eigvecs, eigvals] =
            compute_eigenvectors_on_host(q, std::move(corr), component_count, { corr_event });
        if (desc.get_result_options().test(result_options::eigenvalues)) {
            result.set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count));
        }

        sycl::event sign_flip_event;
        if (desc.get_deterministic()) {
            //eigvecs.to_device(q_);
            sign_flip_event = pr::sign_flip(q, eigvecs, { corr_event });
            //eigvecs.to_host(q_);
        }
        if (desc.get_result_options().test(result_options::eigenvectors)) {
            const auto model = model_t{}.set_eigenvectors(
                homogen_table::wrap(eigvecs.flatten(q, { sign_flip_event }),
                                    component_count,
                                    column_count));
            result.set_model(model);
        }
    }

    return result;
}

template <typename Float>
struct finalize_train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const detail::train_parameters<task::dim_reduction>& params,
                        const input_t& input) const {
        return train<Float, task::dim_reduction>(ctx, desc, params, input);
    }
};

template struct finalize_train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct finalize_train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

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

#include "oneapi/dal/algo/covariance/backend/gpu/finalize_compute_kernel.hpp"

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task::compute>;

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    auto& q = ctx.get_queue();

    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();
    const std::int64_t component_count = input.get_partial_crossproduct().get_column_count();

    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    auto bias = desc.get_bias();
    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    sycl::event event;

    const auto nobs_host = pr::table2ndarray<Float>(q, input.get_partial_n_rows());
    auto rows_count_global = nobs_host.get_data()[0];

    const auto sums =
        pr::table2ndarray_1d<Float>(q, input.get_partial_sum(), sycl::usm::alloc::device);
    const auto xtx =
        pr::table2ndarray<Float>(q, input.get_partial_crossproduct(), sycl::usm::alloc::device);

    sycl::event matrix_event;
    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto cov = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
        sycl::event copy_event;
        {
            ONEDAL_PROFILER_TASK(copy_cov, q);
            copy_event = copy(q, cov, xtx, {});
        }

        {
            ONEDAL_PROFILER_TASK(compute_covariance_matrix, q);
            matrix_event = pr::covariance(q, rows_count_global, sums, cov, bias, { copy_event });
        }
        result.set_cov_matrix(
            (homogen_table::wrap(cov.flatten(q, { matrix_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

        sycl::event copy_corr_event;
        {
            ONEDAL_PROFILER_TASK(copy_corr, q);
            copy_corr_event = copy(q, corr, xtx, {});
        }
        {
            ONEDAL_PROFILER_TASK(compute_covariance_matrix, q);
            matrix_event = pr::correlation(q, rows_count_global, sums, corr, { copy_corr_event });
        }
        result.set_cor_matrix(
            (homogen_table::wrap(corr.flatten(q, { matrix_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::means)) {
        auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
        auto means_event = pr::means(q, rows_count_global, sums, means, { matrix_event });
        result.set_means(homogen_table::wrap(means.flatten(q), 1, column_count));
    }
    return result;
}

template <typename Float>
struct finalize_compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return finalize_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_gpu<float, method::dense, task::compute>;
template struct finalize_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend

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

#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/covariance/backend/gpu/misc.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::covariance::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();

    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();

    auto result = partial_compute_result();
    const auto input_ = input.get_prev();

    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(row_count > 0);
    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);

    auto assume_centered = desc.get_assume_centered();

    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [sums, sums_event] = compute_sums(q, data_nd, assume_centered, {});

    auto [crossproduct, crossproduct_event] = compute_crossproduct(q, data_nd, { sums_event });

    const bool has_nobs_data = input_.get_partial_n_rows().has_data();

    if (has_nobs_data) {
        const auto sums_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_sum(), sycl::usm::alloc::device);
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_n_rows());

        const auto crossproducts_nd = pr::table2ndarray<Float>(q,
                                                               input_.get_partial_crossproduct(),
                                                               sycl::usm::alloc::device);

        auto [result_sums, result_crossproducts, result_nobs, update_event] =
            update_partial_results(q,
                                   crossproduct,
                                   sums,
                                   crossproducts_nd,
                                   sums_nd,
                                   nobs_nd,
                                   row_count,
                                   { crossproduct_event });
        result.set_partial_sum(
            homogen_table::wrap(result_sums.flatten(q, { update_event }), 1, column_count));
        result.set_partial_crossproduct(
            homogen_table::wrap(result_crossproducts.flatten(q, { update_event }),
                                column_count,
                                column_count));
        result.set_partial_n_rows(
            homogen_table::wrap(result_nobs.flatten(q, { update_event }), 1, 1));
    }
    else {
        auto [result_nobs, init_event] = init<Float>(q, row_count, { crossproduct_event });

        result.set_partial_sum(
            homogen_table::wrap(sums.flatten(q, { init_event }), 1, column_count));
        result.set_partial_crossproduct(homogen_table::wrap(crossproduct.flatten(q, { init_event }),
                                                            column_count,
                                                            column_count));
        result.set_partial_n_rows(
            homogen_table::wrap(result_nobs.flatten(q, { init_event }), 1, 1));
    }
    return result;
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method::by_default, task_t> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float, task_t>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method::dense, task_t>;
template struct partial_compute_kernel_gpu<double, method::dense, task_t>;

} // namespace oneapi::dal::covariance::backend

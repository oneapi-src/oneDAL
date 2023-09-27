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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/partial_compute_kernel.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"

#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::basic_statistics::backend {
namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto partial_computation(sycl::queue& q,
                         const pr::ndview<Float, 2>& data,
                         const table weights,
                         const partial_compute_result<task_t>& input_,
                         std::int64_t column_count,
                         std::int64_t row_count,
                         bool weights_enabling,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(init_partial_results, q);

    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);
    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto result_max = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_max_ptr = result_max.get_mutable_data();

    auto result_min = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_min_ptr = result_min.get_mutable_data();

    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums_ptr = result_sums.get_mutable_data();

    auto result_sums2 = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums2_ptr = result_sums2.get_mutable_data();

    auto result_sums2cent = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto data_ptr = data.get_data();
    auto weights_nd = pr::table2ndarray_1d<Float>(q, weights, sycl::usm::alloc::device);
    auto weights_ptr = weights_nd.get_data();

    const Float global_max = de::limits<Float>::max();

    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range(column_count);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            if (id == 0) {
                result_nobs_ptr[0] = row_count;
            }
            result_min_ptr[id] = global_max;
            result_max_ptr[id] = -global_max;
            result_sums_ptr[id] = 0;
            result_sums2_ptr[id] = 0;
            for (std::int64_t row = 0; row < row_count; row++) {
                Float val = data_ptr[id + row * column_count];
                if (weights_enabling) {
                    val *= weights_ptr[row];
                }
                result_max_ptr[id] = sycl::max<Float>(result_max_ptr[id], val);
                result_min_ptr[id] = sycl::min<Float>(result_min_ptr[id], val);
                result_sums_ptr[id] += val;
                result_sums2_ptr[id] += val * val;
            }
        });
    });

    if (!input_.get_nobs().has_data()) {
        return std::make_tuple(result_min,
                               result_max,
                               result_sums,
                               result_sums2,
                               result_sums2cent,
                               result_nobs,
                               update_event);
    }
    else {
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input_.get_nobs());
        const auto min_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_min(), sycl::usm::alloc::device);
        const auto max_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_max());
        const auto sums_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_sum(), sycl::usm::alloc::device);

        const auto sums2_nd = pr::table2ndarray_1d<Float>(q,
                                                          input_.get_partial_sum_squares(),
                                                          sycl::usm::alloc::device);
        const auto sums2cent_nd =
            pr::table2ndarray_1d<Float>(q,
                                        input_.get_partial_sum_squares_centered(),
                                        sycl::usm::alloc::device);

        auto prev_nobs = nobs_nd.get_mutable_data();
        auto prev_min_ptr = min_nd.get_mutable_data();
        auto prev_max_ptr = max_nd.get_mutable_data();
        auto prev_sums_ptr = sums_nd.get_mutable_data();
        auto prev_sums2_ptr = sums2_nd.get_mutable_data();
        auto result_sums2cent_ptr = result_sums2cent.get_mutable_data();
        auto init_event = q.submit([&](sycl::handler& cgh) {
            const auto range = sycl::range<1>(1);

            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::item<1> id) {
                result_nobs_ptr[0] += prev_nobs[0];
            });
        });
        auto merge_event = q.submit([&](sycl::handler& cgh) {
            const auto range = sycl::range<1>(column_count);

            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::item<1> id) {
                result_min_ptr[id] = sycl::fmin(prev_min_ptr[id], result_min_ptr[id]);
                result_max_ptr[id] = sycl::fmax(prev_max_ptr[id], result_max_ptr[id]);

                result_sums_ptr[id] = prev_sums_ptr[id] + result_sums_ptr[id];

                result_sums2_ptr[id] = prev_sums2_ptr[id] + result_sums2_ptr[id];

                result_sums2cent_ptr[id] = result_sums2_ptr[id] - result_sums_ptr[id] *
                                                                      result_sums_ptr[id] /
                                                                      result_nobs_ptr[0];
            });
        });
        return std::make_tuple(result_min,
                               result_max,
                               result_sums,
                               result_sums2,
                               result_sums2cent,
                               result_nobs,
                               merge_event);
    }
}

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();

    const bool weights_enabling = input.get_weights().has_data();
    const auto weights = input.get_weights();

    auto result = partial_compute_result();
    const auto input_ = input.get_prev();
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [result_min,
          result_max,
          result_sums,
          result_sums2,
          result_sums2cent,
          partial_nobs,
          update_event] =
        partial_computation(q, data_nd, weights, input_, column_count, row_count, weights_enabling);

    result.set_partial_min(
        (homogen_table::wrap(result_min.flatten(q, { update_event }), 1, column_count)));
    result.set_partial_max(
        (homogen_table::wrap(result_max.flatten(q, { update_event }), 1, column_count)));

    result.set_partial_sum(
        (homogen_table::wrap(result_sums.flatten(q, { update_event }), 1, column_count)));
    result.set_partial_sum_squares(
        (homogen_table::wrap(result_sums2.flatten(q, { update_event }), 1, column_count)));
    result.set_partial_sum_squares_centered(
        (homogen_table::wrap(result_sums2cent.flatten(q, { update_event }), 1, column_count)));
    result.set_nobs((homogen_table::wrap(partial_nobs.flatten(q, { update_event }), 1, 1)));

    return result;
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method::dense, task::compute>;
template struct partial_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::basic_statistics::backend

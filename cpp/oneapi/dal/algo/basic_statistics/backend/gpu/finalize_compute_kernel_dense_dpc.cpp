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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel.hpp"

#include "oneapi/dal/backend/primitives/reduction.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task::compute>;

//TODO: fix naming+potential performance improvements+if depends by result option
template <typename Float>
auto compute_all_metrics(sycl::queue& q,
                         const pr::ndview<Float, 1>& sums,
                         const pr::ndview<Float, 1>& sums2,
                         const pr::ndview<Float, 1>& sums2cent,
                         const pr::ndview<Float, 1>& nobs,
                         std::size_t column_count,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_all_metrics, q);
    auto result_means = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_variance = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_raw_moment = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_variation = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_stddev = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto result_means_ptr = result_means.get_mutable_data();
    auto result_variance_ptr = result_variance.get_mutable_data();
    auto result_raw_moment_ptr = result_raw_moment.get_mutable_data();
    auto result_variation_ptr = result_variation.get_mutable_data();
    auto result_stddev_ptr = result_stddev.get_mutable_data();

    auto nobs_ptr = nobs.get_data();

    auto sums_data = sums.get_data();
    auto sums2_data = sums2.get_data();
    auto sums2cent_data = sums2cent.get_data();
    const Float inv_n = Float(1.0 / double(nobs_ptr[0]));
    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<1>(column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_means_ptr[id] = sums_data[id] / nobs_ptr[0];
            result_variance_ptr[id] = sums2cent_data[id] / (nobs_ptr[0] - 1);

            result_raw_moment_ptr[id] = sums2_data[id] * inv_n;

            result_stddev_ptr[id] = sycl::sqrt(result_variance_ptr[id]);

            result_variation_ptr[id] = result_stddev_ptr[id] / result_means_ptr[id];
        });
    });

    return std::make_tuple(result_means,
                           result_variance,
                           result_raw_moment,
                           result_variation,
                           result_stddev,
                           update_event);
}

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    auto& q_ = ctx.get_queue();
    result_t res;

    auto column_count = input.get_partial_sum_squares().get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    sycl::event event;
    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    const auto sums_nd =
        pr::table2ndarray_1d<Float>(q_, input.get_partial_sum(), sycl::usm::alloc::device);
    const auto nobs_nd = pr::table2ndarray_1d<Float>(q_, input.get_nobs());

    const auto sums2_nd =
        pr::table2ndarray_1d<Float>(q_, input.get_partial_sum_squares(), sycl::usm::alloc::device);
    const auto sums2cent_nd = pr::table2ndarray_1d<Float>(q_,
                                                          input.get_partial_sum_squares_centered(),
                                                          sycl::usm::alloc::device);
    if (res_op.test(result_options::min)) {
        ONEDAL_ASSERT(input.get_partial_min().get_column_count() == column_count);
        res.set_min(input.get_partial_min());
    }
    if (res_op.test(result_options::max)) {
        ONEDAL_ASSERT(input.get_partial_max().get_column_count() == column_count);
        res.set_max(input.get_partial_max());
    }
    if (res_op.test(result_options::sum)) {
        ONEDAL_ASSERT(input.get_partial_sum().get_column_count() == column_count);
        res.set_sum(input.get_partial_sum());
    }
    if (res_op.test(result_options::sum_squares)) {
        ONEDAL_ASSERT(input.get_partial_sum_squares().get_column_count() == column_count);
        res.set_sum_squares(input.get_partial_sum_squares());
    }

    if (res_op.test(result_options::sum_squares_centered)) {
        ONEDAL_ASSERT(input.get_partial_sum_squares_centered().get_column_count() == column_count);
        res.set_sum_squares_centered(input.get_partial_sum_squares_centered());
    }

    auto [result_means,
          result_variance,
          result_raw_moment,
          result_variation,
          result_stddev,
          update_event] =
        compute_all_metrics<Float>(q_, sums_nd, sums2_nd, sums2cent_nd, nobs_nd, column_count, {});
    if (res_op.test(result_options::mean)) {
        ONEDAL_ASSERT(result_means.get_dimension(0) == column_count);
        res.set_mean(homogen_table::wrap(result_means.flatten(q_, { event }), 1, column_count));
    }
    if (res_op.test(result_options::second_order_raw_moment)) {
        ONEDAL_ASSERT(result_raw_moment.get_dimension(0) == column_count);
        res.set_second_order_raw_moment(
            homogen_table::wrap(result_raw_moment.flatten(q_, { event }), 1, column_count));
    }
    if (res_op.test(result_options::variance)) {
        ONEDAL_ASSERT(result_variance.get_dimension(0) == column_count);
        res.set_variance(
            homogen_table::wrap(result_variance.flatten(q_, { event }), 1, column_count));
    }
    if (res_op.test(result_options::standard_deviation)) {
        ONEDAL_ASSERT(result_stddev.get_dimension(0) == column_count);
        res.set_standard_deviation(
            homogen_table::wrap(result_stddev.flatten(q_, { event }), 1, column_count));
    }
    if (res_op.test(result_options::variation)) {
        ONEDAL_ASSERT(result_variation.get_dimension(0) == column_count);
        res.set_variation(
            homogen_table::wrap(result_variation.flatten(q_, { event }), 1, column_count));
    }
    return res;
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

} // namespace oneapi::dal::basic_statistics::backend

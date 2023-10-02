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
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

//TODO:: optimize
template <typename Float>
auto update_partial_results(sycl::queue& q,
                            const pr::ndview<Float, 1>& min,
                            const pr::ndview<Float, 1>& current_min,
                            const pr::ndview<Float, 1>& max,
                            const pr::ndview<Float, 1>& current_max,
                            const pr::ndview<Float, 1>& sums,
                            const pr::ndview<Float, 1>& current_sums,
                            const pr::ndview<Float, 1>& sums2,
                            const pr::ndview<Float, 1>& current_sums2,
                            const pr::ndview<Float, 1>& sums2cent,
                            const pr::ndview<Float, 1>& current_sums2cent,
                            const std::int64_t column_count,
                            const std::int64_t row_count,
                            const pr::ndview<Float, 1>& nobs,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_results, q);

    auto result_min = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_max = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums2 = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums2cent = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto result_min_ptr = result_min.get_mutable_data();
    auto result_max_ptr = result_max.get_mutable_data();
    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_sums2_ptr = result_sums2.get_mutable_data();
    auto result_sums2cent_ptr = result_sums2cent.get_mutable_data();

    auto current_min_ptr = current_min.get_mutable_data();
    auto current_max_ptr = current_max.get_mutable_data();
    auto current_sums_ptr = current_sums.get_mutable_data();
    auto current_sums2_ptr = current_sums2.get_mutable_data();

    auto nobs_ptr = nobs.get_data();
    auto min_data = min.get_data();
    auto max_data = max.get_data();
    auto sums_data = sums.get_data();
    auto sums2_data = sums2.get_data();

    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<1>(column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_min_ptr[id] = sycl::fmin(current_min_ptr[id], min_data[id]);
            result_max_ptr[id] = sycl::fmax(current_max_ptr[id], max_data[id]);

            result_sums_ptr[id] = current_sums_ptr[id] + sums_data[id];

            result_sums2_ptr[id] = current_sums2_ptr[id] + sums2_data[id];

            result_sums2cent_ptr[id] =
                result_sums2_ptr[id] - result_sums_ptr[id] * result_sums_ptr[id] / nobs_ptr[0];
        });
    });
    return std::make_tuple(result_min,
                           result_max,
                           result_sums,
                           result_sums2,
                           result_sums2cent,
                           update_event);
}

template <typename Float>
auto apply_weights(sycl::queue& q,
                   const pr::ndview<Float, 2>& data,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const pr::ndview<Float, 1>& weights,
                   const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(apply_weights, q);
    auto data_to_compute =
        pr::ndarray<Float, 2>::empty(q, { row_count, column_count }, alloc::device);

    auto weights_ptr = weights.get_data();

    auto data_to_compute_ptr = data_to_compute.get_mutable_data();

    auto input_data = data.get_data();

    auto apply_weights_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<2>(row_count, column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            data_to_compute_ptr[id[0] * column_count + id[1]] =
                input_data[id[0] * column_count + id[1]] * weights_ptr[id[0]];
        });
    });

    return std::make_tuple(data_to_compute, apply_weights_event);
}

//TODO:: create a task realted to investigation bug in reduce_by_columns_function
template <typename Float>
auto init_computation(sycl::queue& q,
                      const pr::ndview<Float, 2>& data,
                      const pr::ndview<Float, 1>& nobs,
                      std::int64_t column_count,
                      std::int64_t row_count,
                      const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(init_partial_results, q);

    auto component_count = column_count;
    auto current_nobs_ptr = nobs.get_data();
    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);
    auto result_nobs_ptr = result_nobs.get_mutable_data();
    auto result_max = pr::ndarray<Float, 1>::empty(q, component_count, alloc::device);

    auto result_min = pr::ndarray<Float, 1>::empty(q, component_count, alloc::device);

    auto result_sums = pr::ndarray<Float, 1>::empty(q, component_count, alloc::device);

    auto result_sums2 = pr::ndarray<Float, 1>::empty(q, component_count, alloc::device);

    auto result_sums2cent = pr::ndarray<Float, 1>::empty(q, component_count, alloc::device);

    auto nobs_update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range(1);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_nobs_ptr[0] = current_nobs_ptr[0] + row_count;
        });
    });
    auto reduce_event_min = pr::reduce_by_columns(q,
                                                  data,
                                                  result_min,
                                                  pr::min<Float>{},
                                                  pr::identity<Float>{},
                                                  { nobs_update_event });
    reduce_event_min.wait_and_throw();
    auto reduce_event_max = pr::reduce_by_columns(q,
                                                  data,
                                                  result_max,
                                                  pr::max<Float>{},
                                                  pr::identity<Float>{},
                                                  { reduce_event_min });
    reduce_event_max.wait_and_throw();
    auto reduce_event_sums = pr::reduce_by_columns(q,
                                                   data,
                                                   result_sums,
                                                   pr::sum<Float>{},
                                                   pr::identity<Float>{},
                                                   { reduce_event_min });
    reduce_event_sums.wait_and_throw();
    auto reduce_event_sumssquares = pr::reduce_by_columns(q,
                                                          data,
                                                          result_sums2,
                                                          pr::sum<Float>{},
                                                          pr::square<Float>{},
                                                          { reduce_event_min });
    reduce_event_sumssquares.wait_and_throw();

    return std::make_tuple(result_min,
                           result_max,
                           result_sums,
                           result_sums2,
                           result_sums2cent,
                           result_nobs,
                           reduce_event_sumssquares);
}

//TODO: add checks
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
    const std::int64_t component_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto data_to_compute = data_nd;
    sycl::event apply_weights_event;
    if (weights_enabling) {
        auto weights_nd = pr::table2ndarray_1d<Float>(q, weights, sycl::usm::alloc::device);
        std::tie(data_to_compute, apply_weights_event) =
            apply_weights(q, data_nd, row_count, column_count, weights_nd);
    }

    const bool has_nobs_data = input_.get_partial_n_rows().has_data();

    if (has_nobs_data) {
        const auto sums_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_sum(), sycl::usm::alloc::device);
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_n_rows());

        const auto min_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_min(), sycl::usm::alloc::device);
        const auto max_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_max());

        const auto sums2_nd = pr::table2ndarray_1d<Float>(q,
                                                          input_.get_partial_sum_squares(),
                                                          sycl::usm::alloc::device);
        const auto sums2cent_nd =
            pr::table2ndarray_1d<Float>(q,
                                        input_.get_partial_sum_squares_centered(),
                                        sycl::usm::alloc::device);
        auto [partial_min,
              partial_max,
              partial_sums,
              partial_sums2,
              partial_sums2cent,
              partial_nobs,
              init_computation_event] = init_computation(q,
                                                         data_to_compute,
                                                         nobs_nd,
                                                         column_count,
                                                         row_count,
                                                         { apply_weights_event });

        auto [result_min,
              result_max,
              result_sums,
              result_sums2,
              result_sums2cent,
              merge_results_event] = update_partial_results(q,
                                                            min_nd,
                                                            partial_min,
                                                            max_nd,
                                                            partial_max,
                                                            sums_nd,
                                                            partial_sums,
                                                            sums2_nd,
                                                            partial_sums2,
                                                            sums2cent_nd,
                                                            partial_sums2cent,
                                                            column_count,
                                                            row_count,
                                                            partial_nobs,
                                                            { init_computation_event });
        result.set_partial_min(
            (homogen_table::wrap(result_min.flatten(q, { merge_results_event }), 1, column_count)));
        result.set_partial_max(
            (homogen_table::wrap(result_max.flatten(q, { merge_results_event }), 1, column_count)));

        result.set_partial_sum((
            homogen_table::wrap(result_sums.flatten(q, { merge_results_event }), 1, column_count)));
        result.set_partial_sum_squares(
            (homogen_table::wrap(result_sums2.flatten(q, { merge_results_event }),
                                 1,
                                 column_count)));
        result.set_partial_sum_squares_centered(
            (homogen_table::wrap(result_sums2cent.flatten(q, { merge_results_event }),
                                 1,
                                 column_count)));
        result.set_partial_n_rows(
            (homogen_table::wrap(partial_nobs.flatten(q, { merge_results_event }), 1, 1)));
    }
    else {
        auto init_nobs = pr::ndarray<Float, 1>::empty(q, 1);

        auto [result_min,
              result_max,
              result_sums,
              result_sums2,
              result_sums2cent,
              result_nobs,
              init_computation_event] = init_computation(q,
                                                         data_to_compute,
                                                         init_nobs,
                                                         column_count,
                                                         row_count,
                                                         { apply_weights_event });

        result.set_partial_min(
            (homogen_table::wrap(result_min.flatten(q, { init_computation_event }),
                                 1,
                                 column_count)));
        result.set_partial_max(
            (homogen_table::wrap(result_max.flatten(q, { init_computation_event }),
                                 1,
                                 column_count)));
        result.set_partial_sum(
            (homogen_table::wrap(result_sums.flatten(q, { init_computation_event }),
                                 1,
                                 column_count)));
        result.set_partial_sum_squares(
            (homogen_table::wrap(result_sums2.flatten(q, { init_computation_event }),
                                 1,
                                 column_count)));
        result.set_partial_sum_squares_centered(
            (homogen_table::wrap(result_sums2cent.flatten(q, { init_computation_event }),
                                 1,
                                 column_count)));
        result.set_partial_n_rows(
            (homogen_table::wrap(result_nobs.flatten(q, { init_computation_event }), 1, 1)));
    }

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

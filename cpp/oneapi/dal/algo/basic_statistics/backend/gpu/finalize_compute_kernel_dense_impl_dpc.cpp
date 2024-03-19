/*******************************************************************************
* Copyright contributors to the oneDAL project
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
#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task::compute>;

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

///  A wrapper that computes arrays of basic statistics
///  The choice is based on the optional results
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  desc  The descriptor of the algorithm
/// @param[in]  input The partial_compute_result class with partial sums and xtx matrix
///
/// @return The compute_result object, which contains functions to get covariance/correlation matrix or means.
template <typename Float>
result_t finalize_compute_kernel_dense_impl<Float>::operator()(const descriptor_t& desc,
                                                               const input_t& input) {
    result_t res;
    auto local_desc = get_desc_to_compute<Float>(desc);
    const auto res_op_partial = local_desc.get_result_options();
    auto column_count = 0;

    if (res_op_partial.test(result_options::min)) {
        column_count = input.get_partial_min().get_column_count();
    }
    if (res_op_partial.test(result_options::sum)) {
        column_count = input.get_partial_sum().get_column_count();
    }

    ONEDAL_ASSERT(column_count > 0);

    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input.get_partial_n_rows());

    if (res_op.test(result_options::min)) {
        ONEDAL_ASSERT(input.get_partial_min().get_column_count() == column_count);
        const auto min =
            pr::table2ndarray_1d<Float>(q, input.get_partial_min(), sycl::usm::alloc::device);

        { comm_.allreduce(min.flatten(q, {}), spmd::reduce_op::min).wait(); }
        res.set_min(homogen_table::wrap(min.flatten(q, {}), 1, column_count));
    }

    if (res_op.test(result_options::max)) {
        ONEDAL_ASSERT(input.get_partial_max().get_column_count() == column_count);
        const auto max =
            pr::table2ndarray_1d<Float>(q, input.get_partial_max(), sycl::usm::alloc::device);

        { comm_.allreduce(max.flatten(q, {}), spmd::reduce_op::max).wait(); }
        res.set_max(homogen_table::wrap(max.flatten(q, {}), 1, column_count));
    }

    if (res_op_partial.test(result_options::sum)) {
        const auto sums_nd =
            pr::table2ndarray_1d<Float>(q, input.get_partial_sum(), sycl::usm::alloc::device);
        {
            ONEDAL_PROFILER_TASK(allreduce_sums, q);
            comm_.allreduce(sums_nd.flatten(q, {}), spmd::reduce_op::sum).wait();
        }
        const auto sums2_nd = pr::table2ndarray_1d<Float>(q,
                                                          input.get_partial_sum_squares(),
                                                          sycl::usm::alloc::device);
        {
            ONEDAL_PROFILER_TASK(allreduce_sums, q);
            comm_.allreduce(sums2_nd.flatten(q, {}), spmd::reduce_op::sum).wait();
        }
        const auto sums2cent_nd =
            pr::table2ndarray_1d<Float>(q,
                                        input.get_partial_sum_squares_centered(),
                                        sycl::usm::alloc::device);
        {
            ONEDAL_PROFILER_TASK(allreduce_sums, q);
            comm_.allreduce(sums2cent_nd.flatten(q, {}), spmd::reduce_op::sum).wait();
        }
        auto [result_means,
              result_variance,
              result_raw_moment,
              result_variation,
              result_stddev,
              update_event] = compute_all_metrics<Float>(q,
                                                         sums_nd,
                                                         sums2_nd,
                                                         sums2cent_nd,
                                                         nobs_nd,
                                                         column_count,
                                                         {});

        if (res_op.test(result_options::sum)) {
            ONEDAL_ASSERT(input.get_partial_sum().get_column_count() == column_count);
            res.set_sum(input.get_partial_sum());
        }

        if (res_op.test(result_options::sum_squares)) {
            ONEDAL_ASSERT(input.get_partial_sum_squares().get_column_count() == column_count);
            res.set_sum_squares(input.get_partial_sum_squares());
        }

        if (res_op.test(result_options::sum_squares_centered)) {
            ONEDAL_ASSERT(input.get_partial_sum_squares_centered().get_column_count() ==
                          column_count);
            res.set_sum_squares_centered(input.get_partial_sum_squares_centered());
        }

        if (res_op.test(result_options::mean)) {
            ONEDAL_ASSERT(result_means.get_dimension(0) == column_count);
            res.set_mean(
                homogen_table::wrap(result_means.flatten(q, { update_event }), 1, column_count));
        }

        if (res_op.test(result_options::second_order_raw_moment)) {
            ONEDAL_ASSERT(result_raw_moment.get_dimension(0) == column_count);
            res.set_second_order_raw_moment(
                homogen_table::wrap(result_raw_moment.flatten(q, { update_event }),
                                    1,
                                    column_count));
        }

        if (res_op.test(result_options::variance)) {
            ONEDAL_ASSERT(result_variance.get_dimension(0) == column_count);
            res.set_variance(
                homogen_table::wrap(result_variance.flatten(q, { update_event }), 1, column_count));
        }

        if (res_op.test(result_options::standard_deviation)) {
            ONEDAL_ASSERT(result_stddev.get_dimension(0) == column_count);
            res.set_standard_deviation(
                homogen_table::wrap(result_stddev.flatten(q, { update_event }), 1, column_count));
        }

        if (res_op.test(result_options::variation)) {
            ONEDAL_ASSERT(result_variation.get_dimension(0) == column_count);
            res.set_variation(homogen_table::wrap(result_variation.flatten(q, { update_event }),
                                                  1,
                                                  column_count));
        }
    }
    return res;
}

template class finalize_compute_kernel_dense_impl<float>;
template class finalize_compute_kernel_dense_impl<double>;

} // namespace oneapi::dal::basic_statistics::backend
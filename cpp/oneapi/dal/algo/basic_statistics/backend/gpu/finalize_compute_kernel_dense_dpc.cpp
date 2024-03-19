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

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    return finalize_compute_kernel_dense_impl<Float>(ctx)(desc, input);
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

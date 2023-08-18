/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/regression/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename DistType, typename RespType>
sycl::event distance_regression_kernel(sycl::queue& queue,
                                       const ndview<RespType, 2>& responses,
                                       const ndview<DistType, 2>& distances,
                                       ndview<RespType, 1>& result,
                                       const event_vector& deps) {
    constexpr auto eps = dal::detail::limits<DistType>::epsilon();
    ONEDAL_ASSERT(distances.has_data());
    ONEDAL_ASSERT(responses.has_data());
    ONEDAL_ASSERT(result.has_mutable_data());
    const auto samples = dal::detail::integral_cast<std::int32_t>(responses.get_dimension(0));
    const auto k_resps = dal::detail::integral_cast<std::int32_t>(responses.get_dimension(1));
    ONEDAL_ASSERT(samples == result.get_dimension(0));
    ONEDAL_ASSERT(samples == distances.get_dimension(0));
    ONEDAL_ASSERT(k_resps == distances.get_dimension(1));
    const auto* const inp_ptr = responses.get_data();
    const auto* const dst_ptr = distances.get_data();
    auto* const res_ptr = result.get_mutable_data();
    const auto inp_str = responses.get_leading_stride();
    const auto dst_str = distances.get_leading_stride();
    const auto range = make_range_1d(samples);
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> item) {
            const auto* const dst_row = dst_ptr + item * dst_str;
            const auto* const inp_row = inp_ptr + item * inp_str;
            DistType denom = 0;
            for (std::int32_t i = 0; i < k_resps; ++i) {
                const auto& dst = dst_row[i];
                denom += (dst < eps) ? 1 : (1 / dst);
            }
            RespType proto = 0;
            for (std::int32_t i = 0; i < k_resps; ++i) {
                const auto& dst = dst_row[i];
                const auto weight = (dst < eps) ? 1 : (1 / dst);
                proto += (weight * inp_row[i]);
            }
            res_ptr[item] = proto / denom;
        });
    });
}

template <typename DistType, typename ResponseType>
naive_distance_regression<DistType, ResponseType>::naive_distance_regression(sycl::queue& queue)
        : base_t(queue) {}

template <typename DistType, typename ResponseType>
sycl::event naive_distance_regression<DistType, ResponseType>::operator()(
    const ndview<ResponseType, 2>& responses,
    const ndview<DistType, 2>& distances,
    ndview<ResponseType, 1>& results,
    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(regression.distance, this->get_queue());
    return distance_regression_kernel(this->get_queue(), responses, distances, results, deps);
}

#define INSTANTIATE(F, R) template class naive_distance_regression<F, R>;

INSTANTIATE(float, float);
INSTANTIATE(float, double);
INSTANTIATE(double, float);
INSTANTIATE(double, double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives

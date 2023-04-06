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

#include "oneapi/dal/backend/primitives/voting/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename DistsType, typename IndexType>
sycl::event distance_voting_kernel(sycl::queue& queue,
                                   const ndview<IndexType, 2>& responses,
                                   const ndview<DistsType, 2>& distances,
                                   ndview<DistsType, 2>& probas,
                                   ndview<IndexType, 1>& result,
                                   const event_vector& deps) {
    constexpr auto eps = dal::detail::limits<DistsType>::epsilon();
    ONEDAL_ASSERT(distances.has_data());
    ONEDAL_ASSERT(responses.has_data());
    ONEDAL_ASSERT(result.has_mutable_data());
    ONEDAL_ASSERT(probas.has_mutable_data());
    const auto classes = dal::detail::integral_cast<std::int32_t>(probas.get_dimension(1));
    const auto samples = dal::detail::integral_cast<std::int32_t>(responses.get_dimension(0));
    const auto k_resps = dal::detail::integral_cast<std::int32_t>(responses.get_dimension(1));
    ONEDAL_ASSERT(samples >= probas.get_dimension(0));
    ONEDAL_ASSERT(samples == result.get_dimension(0));
    ONEDAL_ASSERT(samples == distances.get_dimension(0));
    ONEDAL_ASSERT(k_resps == distances.get_dimension(1));
    const auto* const ids_ptr = responses.get_data();
    const auto* const dst_ptr = distances.get_data();
    auto* const prb_ptr = probas.get_mutable_data();
    auto* const res_ptr = result.get_mutable_data();
    const auto prb_str = probas.get_leading_stride();
    const auto ids_str = responses.get_leading_stride();
    const auto dst_str = distances.get_leading_stride();
    const auto range = make_range_1d(samples);
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> item) {
            auto* const prb_row = prb_ptr + item * prb_str;
            const auto* const dst_row = dst_ptr + item * dst_str;
            const auto* const ids_row = ids_ptr + item * ids_str;
            for (std::int32_t i = 0; i < classes; ++i) {
                prb_row[i] = 0;
            }
            bool contains_zero = false;
            for (std::int32_t i = 0; i < k_resps; ++i) {
                const auto dst = dst_row[i];
                if (dst < eps) {
                    contains_zero = true;
                    break;
                }
            }

            if (contains_zero) {
                for (std::int32_t i = 0; i < k_resps; ++i) {
                    const auto dst = dst_row[i];
                    const auto idx = ids_row[i];
                    prb_row[idx] = dst < eps ? 1 : 0;
                }
            }
            else {
                for (std::int32_t i = 0; i < k_resps; ++i) {
                    const auto dst = dst_row[i];
                    const auto idx = ids_row[i];
                    prb_row[idx] += (dst < eps) ? 1 : (1 / dst);
                }
            }

            IndexType best_cls = -1;
            DistsType best_prb = -1;
            for (std::int32_t i = 0; i < classes; ++i) {
                const auto p = prb_row[i];
                const bool handle = p > best_prb;
                best_cls = handle ? i : best_cls;
                best_prb = handle ? p : best_prb;
            }
            res_ptr[item] = best_cls;
        });
    });
}

template <typename DistType, typename ClassType>
naive_distance_voting<DistType, ClassType>::naive_distance_voting(sycl::queue& queue,
                                                                  std::int64_t max_block,
                                                                  std::int64_t class_count)
        : base_t(queue, class_count),
          global_probas_(ndarray<DistType, 2>::empty(queue,
                                                     { max_block, class_count },
                                                     sycl::usm::alloc::device)) {}

template <typename DistType, typename ClassType>
ndview<DistType, 2>& naive_distance_voting<DistType, ClassType>::get_global_probas() {
    return global_probas_;
}

template <typename DistType, typename ClassType>
sycl::event naive_distance_voting<DistType, ClassType>::operator()(
    const ndview<ClassType, 2>& responses,
    const ndview<DistType, 2>& distances,
    ndview<ClassType, 1>& results,
    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(voting.distance, this->get_queue());

    const auto samples_count = results.get_dimension(0);
    auto p_slice = this->get_global_probas().get_row_slice(0, samples_count);
    return distance_voting_kernel(this->get_queue(), responses, distances, p_slice, results, deps);
}

#define INSTANTIATE(F, C) template class naive_distance_voting<F, C>;

INSTANTIATE(float, std::int32_t);
INSTANTIATE(float, std::int64_t);
INSTANTIATE(double, std::int32_t);
INSTANTIATE(double, std::int64_t);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives

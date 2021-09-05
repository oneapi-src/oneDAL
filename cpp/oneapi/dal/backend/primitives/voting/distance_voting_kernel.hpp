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

#pragma once

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template<typename DistsType, typename IndexType>
inline sycl::event distance_voting_kernel(sycl::queue& queue,
                                          std::int64_t class_count,
                                          const ndview<IndexType>& responses,
                                          const ndview<DistsType>& distances,
                                          ndview<IndexType>& results,
                                          const event_vector& deps = {}) {
    ONEDAL_ASSERT(distances.has_data());
    ONEDAL_ASSERT(responses.has_data());
    ONEDAL_ASSERT(results.has_mutable_data());
    const std::int32_t r = responses.get_dimension(0);
    const std::int32_t k = responses.get_dimension(1);
    ONEDAL_ASSERT(r == distances.get_dimension(0));
    ONEDAL_ASSERT(k == distances.get_dimension(1));
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(ndrange,[=](sycl::nd_item<1> item) {
            auto sg = item.get_sub_group();

            const std::int32_t
        });
    });
}


#endif

} // namespace oneapi::dal::backend::primitives

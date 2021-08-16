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

#include "oneapi/dal/backend/primitives/voting/uniform.hpp"

namespace oneapi::dal::backend::primitives {

/*template<typename ClasssType = std::int32_t, std::int32_t c_max = 16>
sycl::event small_k_uniform_vote(sycl::queue& q,
                                 const ndview<ClassType, 2>& responses,
                                 ndview<ClassType, 1>& results,
                                 std::int64_t
                                 const event_vector& deps = {}) {
    static_assert(k_max > 0);
    ONEDAL_ASSERT(responses.has_data());
    ONEDAL_ASSERT(results.has_mutable_data());
    ONEDAL_ASSERT(k_max >= responses.get_dimension(1));
    ONEDAL_ASSERT(responses.get_dimension(0) == results.get_dimension(0));
    const std::int64_t k_current = responses.get_dimension(1);
    const std::int64_t n_samples = responses.get_dimension(0)
    if (k_max == k_current) {
        const auto range = make_range_1d(n_samples);
        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(range, [=](sycl::id<1> idx) {

            });
        });
    }
    if constexpr (k_max > 1) {
        return small_k_uniform_vote(q,
                                    responses,
                                    results,
                                    deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
}*/


} // namespace oneapi::dal::backend::primitives

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

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

template <typename Float>
inline bool is_upper_edge(const Float y, const Float alpha, const Float C) {
    return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
}

template <typename Float>
inline bool is_lower_edge(const Float y, const Float alpha, const Float C) {
    return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
}

template <typename Float>
inline sycl::event invert_values(sycl::queue& q,
                                 const pr::ndview<Float, 1>& data,
                                 pr::ndview<Float, 1>& res,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(0) == res.get_dimension(0));

    const Float* data_ptr = data.get_data();
    Float* res_ptr = res.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(data.get_count());
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            res_ptr[idx] = -data_ptr[idx];
        });
    });
}

} // namespace oneapi::dal::svm::backend
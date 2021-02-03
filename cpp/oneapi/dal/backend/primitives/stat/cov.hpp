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

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
struct compute_cov_op {
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           const ndview<Float, 1>& sums,
                           const ndview<Float, 2>& cov,
                           const event_vector& deps);
};

template <typename Float>
inline sycl::event compute_cov(sycl::queue& queue,
                               const ndview<Float, 2>& data,
                               const ndview<Float, 1>& sums,
                               const ndview<Float, 2>& cov,
                               const event_vector& deps = {}) {
    return compute_cov_op<Float>{}(queue, data, sums, cov, deps);
}

#endif

} // namespace oneapi::dal::backend::primitives

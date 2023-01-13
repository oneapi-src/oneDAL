/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, ndorder ao>
sycl::event gemv(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 Float alpha = Float(1),
                 Float beta = Float(0),
                 const event_vector& deps = {});

template <typename Float, ndorder ao>
inline sycl::event gemv(sycl::queue& queue,
                        const ndview<Float, 2, ao>& a,
                        const ndview<Float, 1>& x,
                        ndview<Float, 1>& y,
                        const event_vector& deps = {}) {
    return gemv<Float>(queue, a, x, y, Float(1), Float(0), deps);
}

#endif

} // namespace oneapi::dal::backend::primitives

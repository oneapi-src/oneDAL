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
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

namespace mkl = oneapi::fpk;

template <mkl::uplo ul, typename Float, ndorder ao, ndorder co>
sycl::event syrk(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 ndview<Float, 2, co>& c,
                 Float alpha = Float(1),
                 Float beta = Float(0),
                 const event_vector& deps = {});

template <typename Float, ndorder ao, ndorder co>
inline sycl::event syrk(sycl::queue& queue,
                        mkl::uplo uplo,
                        const ndview<Float, 2, ao>& a,
                        ndview<Float, 2, co>& c,
                        Float alpha = Float(1),
                        Float beta = Float(0),
                        const event_vector& deps = {}) {
    sycl::event res_event;
    if (uplo == mkl::uplo::upper) {
        constexpr auto upper = mkl::uplo::upper;
        res_event = syrk<upper>(queue, a, c, alpha, beta, deps);
    } else {
        constexpr auto lower = mkl::uplo::lower;
        res_event = syrk<lower>(queue, a, c, alpha, beta, deps);
    }
    return res_event;
}

template <mkl::uplo ul, typename Float, ndorder ao, ndorder co>
inline sycl::event syrk(sycl::queue& queue,
                        const ndview<Float, 2, ao>& a,
                        ndview<Float, 2, co>& c,
                        const event_vector& deps = {}) {
    return syrk<Float>(queue, a, c, Float(1), Float(0), deps);
}

template <typename Float, ndorder ao, ndorder co>
inline sycl::event syrk(sycl::queue& queue,
                        mkl::uplo uplo,
                        const ndview<Float, 2, ao>& a,
                        ndview<Float, 2, co>& c,
                        const event_vector& deps = {}) {
    return syrk<Float>(queue, uplo, a, c, Float(1), Float(0), deps);
}

#endif

} // namespace oneapi::dal::backend::primitives

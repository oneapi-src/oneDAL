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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::linear_regression::backend {

namespace be = dal::backend;
namespace pr = be::primitives;

template <bool beta, typename Float, pr::ndorder layout>
sycl::event update_xtx(sycl::queue& queue,
                       const pr::ndview<Float, 2, layout>& x,
                       pr::ndview<Float, 2, pr::ndorder::c>& xtx,
                       const be::event_vector& deps = {});

template <typename Float, pr::ndorder layout>
inline sycl::event update_xtx(sycl::queue& queue,
                              bool beta,
                              const pr::ndview<Float, 2, layout>& x,
                              pr::ndview<Float, 2, pr::ndorder::c>& xtx,
                              const be::event_vector& deps = {}) {
    if (beta)
        return update_xtx<true>(queue, x, xtx, deps);
    else
        return update_xtx<false>(queue, x, xtx, deps);
}

template <bool beta, typename Float, pr::ndorder xlayout, pr::ndorder ylayout>
sycl::event update_xty(sycl::queue& queue,
                       const pr::ndview<Float, 2, xlayout>& x,
                       const pr::ndview<Float, 2, ylayout>& y,
                       pr::ndview<Float, 2, pr::ndorder::f>& xty,
                       const be::event_vector& deps = {});

template <typename Float, pr::ndorder xlayout, pr::ndorder ylayout>
inline sycl::event update_xty(sycl::queue& queue,
                              bool beta,
                              const pr::ndview<Float, 2, xlayout>& x,
                              const pr::ndview<Float, 2, ylayout>& y,
                              pr::ndview<Float, 2, pr::ndorder::f>& xty,
                              const be::event_vector& deps = {}) {
    if (beta)
        return update_xty<true>(queue, x, y, xty, deps);
    else
        return update_xty<false>(queue, x, y, xty, deps);
}

} // namespace oneapi::dal::linear_regression::backend

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

#include "oneapi/dal/algo/linear_regression/backend/gpu/update_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

template<bool beta, typename Float, pr::ndorder layout>
sycl::event update_xtx( sycl::queue& queue,
                        const pr::ndview<Float, 2, layout>& x,
                        pr::ndview<Float, 2>& xtx,
                        const be::event_vector& deps) {
    ONEDAL_ASSERT(x.has_data());
    const auto f_count = x.get_dimension(1);
    [[maybe_unused]] const auto s_count = x.get_dimension(0);

    ONEDAL_ASSERT(xtx.has_mutable_data());
    const auto ext_f_count = xtx.get_dimension(1);
    ONEDAL_ASSERT(ext_f_count == xtx.get_dimension(0));
    ONEDAL_ASSERT(ext_f_count == (f_count + std::int64_t(beta)));

    auto [ones, ones_event] = pr::ndarray<Float, 2>::ones(queue,
            {std::int64_t(1), f_count}, sycl::usm::alloc::device);

    auto xtx_core = xtx.get_col_slice(0, f_count).get_row_slice(0, f_count);

    auto syrk_event = pr::syrk(queue, );

    return syrk_event;
}

#define INSTANTIATE(B, F, L)                                    \
template sycl::event update_xtx<B>( sycl::queue&,               \
                                    const pr::ndview<F, 2, L>&, \
                                    pr::ndview<F, 2>&,          \
                                    const be::event_vector&);

#define INSTANTIATE_LAYOUT(B, F)    \
INSTANTIATE(B, F, pr::ndorder::c)   \
INSTANTIATE(B, F, pr::ndorder::f)

#define INSTANTIATE_FLOAT(B)    \
INSTANTIATE_LAYOUT(B, float)    \
INSTANTIATE_LAYOUT(B, double)

INSTANTIATE_FLOAT(true);
INSTANTIATE_FLOAT(false);


} // namespace oneapi::dal::linear_regression::backend

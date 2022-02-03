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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, ndorder order>
sycl::event compute_inversed_l2_norms(sycl::queue& q,
                                      const ndview<Float, 2, order>& inp,
                                      ndview<Float, 1>& out,
                                      const event_vector& deps = {});

template <typename Float, ndorder order>
std::tuple<ndarray<Float, 1>, sycl::event> compute_inversed_l2_norms(
    sycl::queue& q,
    const ndview<Float, 2, order>& inp,
    const event_vector& deps = {},
    const sycl::usm::alloc& alloc = sycl::usm::alloc::device);

template <typename Float, ndorder order1, ndorder order2>
sycl::event compute_cosine_inner_product(sycl::queue& q,
                                         const ndview<Float, 2, order1>& inp1,
                                         const ndview<Float, 2, order2>& inp2,
                                         ndview<Float, 2>& out,
                                         const event_vector& deps = {});

template <typename Float>
sycl::event finalize_cosine(sycl::queue& q,
                            const ndview<Float, 1>& inp1,
                            const ndview<Float, 1>& inp2,
                            ndview<Float, 2>& out,
                            const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives

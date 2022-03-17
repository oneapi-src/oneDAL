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

template<bool beta, typename Float, pr::ndorder xtxlayout, pr::ndorder xtylayout>
sycl::event finalize(   sycl::queue& queue,
                        const pr::ndview<Float, 2, xtxlayout>& xtx,
                        const pr::ndview<Float, 2, xtylayout>& xty,
                        pr::ndview<Float, 2, pr::ndorder::c>& beta,
                        const be::event_vector& deps) {
    return sycl::event{};
}

} // namespace oneapi::dal::linear_regression::backend

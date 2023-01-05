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

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

namespace mkl = oneapi::fpk;

inline constexpr mkl::transpose f_order_as_transposed(ndorder order) {
    return (order == ndorder::f) ? mkl::transpose::trans : mkl::transpose::nontrans;
}

inline constexpr mkl::transpose c_order_as_transposed(ndorder order) {
    return (order == ndorder::c) ? mkl::transpose::trans : mkl::transpose::nontrans;
}

inline constexpr mkl::uplo flip_uplo(mkl::uplo order) {
    constexpr auto upper = mkl::uplo::upper;
    constexpr auto lower = mkl::uplo::lower;
    return (order == upper) ? lower : upper;
}

inline constexpr mkl::uplo ident_uplo(mkl::uplo order) {
    constexpr auto upper = mkl::uplo::upper;
    constexpr auto lower = mkl::uplo::lower;
    return (order == upper) ? upper : lower;
}

} // namespace oneapi::dal::backend::primitives

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

namespace oneapi::dal::backend::primitives::blas {

enum class layout {
    row_major,
    column_major
};

enum class orientation {
    normal,
    transposed
};

template <typename Float>
sycl::event gemm(sycl::queue& q, layout lyt, );

} // namespace oneapi::dal::backend::primitives::blas

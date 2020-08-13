/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <cstdint>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename Float>
struct select_small_k_l2_kernel;

} // impl

template<typename Float>
struct select_small_k_l2;

#endif

} // oneapi::dal::backend::primitives

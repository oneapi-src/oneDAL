/*******************************************************************************
* Copyright contributors to the oneDAL project
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

void uniform_gen_gpu(sycl::queue& queue,
                     std::int64_t count_,
                     int* dst,
                     std::uint8_t* state,
                     int a,
                     int b,
                     const event_vector& deps = {});

void uniform_without_replacement_gen_gpu(sycl::queue& queue,
                                         std::int64_t count_,
                                         int* dst,
                                         std::uint8_t* state,
                                         int a,
                                         int b);

template <typename Float>
void uniform_gen_gpu_float(sycl::queue& queue,
                           std::int64_t count_,
                           Float* dst,
                           std::uint8_t* state,
                           Float a,
                           Float b);

#endif

} // namespace oneapi::dal::backend::primitives

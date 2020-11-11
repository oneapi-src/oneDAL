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

#include "oneapi/dal/detail/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>

namespace oneapi::dal::detail {
namespace v1 {

inline void wait_and_throw(const sycl::vector_class<sycl::event>& dependencies) {
    sycl::event::wait_and_throw(dependencies);
}

} // namespace v1

using v1::wait_and_throw;

} // namespace oneapi::dal::detail

#endif // ONEDAL_DATA_PARALLEL

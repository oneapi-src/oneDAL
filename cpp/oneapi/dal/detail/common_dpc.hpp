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

#ifdef ONEAPI_DAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif // ONEAPI_DAL_DATA_PARALLEL

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

#ifdef ONEAPI_DAL_DATA_PARALLEL
inline void wait_and_throw(const sycl::vector_class<sycl::event>& dependencies) {
    sycl::event::wait_and_throw(dependencies);
}
#endif // ONEAPI_DAL_DATA_PARALLEL

} // namespace oneapi::dal::detail

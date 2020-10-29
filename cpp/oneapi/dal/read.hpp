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

#include "oneapi/dal/detail/read_ops.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal {
namespace v1 {

template <typename Object, typename... Args>
auto read(Args&&... args) {
    return detail::read_dispatch<Object>(std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Object, typename... Args>
auto read(sycl::queue& queue, Args&&... args) {
    return detail::read_dispatch<Object>(detail::data_parallel_policy{ queue },
                                         std::forward<Args>(args)...);
}
#endif

} // namespace v1

using v1::read;

} // namespace oneapi::dal

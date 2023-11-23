/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <sycl/sycl.hpp>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/array_utils.hpp"

template <typename Type>
inline auto to_host(const oneapi::dal::array<Type>& array) {
    oneapi::dal::detail::default_host_policy policy{};
    return oneapi::dal::detail::copy(policy, array);
}

template <typename Type>
inline auto to_device(sycl::queue& queue, const oneapi::dal::array<Type>& array) {
    oneapi::dal::detail::data_parallel_policy policy{ queue };
    return oneapi::dal::detail::copy(policy, array);
}

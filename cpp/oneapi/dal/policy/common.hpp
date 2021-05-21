/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal::preview {

template <typename T>
struct is_execution_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

template <typename T>
struct is_host_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_host_policy_v = is_host_policy<T>::value;

#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
struct is_sycl_policy : std::bool_constant<false> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
constexpr bool is_sycl_policy_v = is_sycl_policy<T>::value;
#endif

template <typename T>
struct is_local_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_local_policy_v = is_local_policy<T>::value;

template <typename T>
struct is_distributed_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_distributed_policy_v = is_distributed_policy<T>::value;

} // namespace oneapi::dal::preview

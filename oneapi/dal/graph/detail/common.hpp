/* file: common.hpp */
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

/// @file
/// Graph related common data type aliases

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::detail {

template <typename IndexType>
constexpr bool is_valid_index_v = dal::detail::is_one_of_v<IndexType, std::int32_t>;

template <typename EdgeValue>
constexpr bool is_valid_edge_value_v =
    dal::detail::is_one_of_v<EdgeValue, empty_value, double, std::int32_t>;

} // namespace oneapi::dal::preview::detail

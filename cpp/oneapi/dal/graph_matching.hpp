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

/// @file
/// Contains the definition of the main processing function for graph matching
/// family of the algorithms

#pragma once

#include "oneapi/dal/detail/graph_matching_ops.hpp"

namespace oneapi::dal::preview {

/// The main processing function for graph matching family of the algorithms
template <typename... Args>
auto graph_matching(Args &&... args) {
    return detail::graph_matching_dispatch(std::forward<Args>(args)...);
}

} // namespace oneapi::dal::preview

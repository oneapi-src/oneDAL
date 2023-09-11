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

/// @file
/// Contains the definition of the input and output for Louvain
/// algorithm

#pragma once

#include "oneapi/dal/algo/louvain/common.hpp"

namespace oneapi::dal::preview::louvain::detail {

class vertex_partitioning_result_impl;

template <typename Graph, typename Task>
class vertex_partitioning_input_impl : public base {
public:
    vertex_partitioning_input_impl(const Graph &g, const table &labels)
            : graph_data(g),
              labels_data(labels) {}

    const Graph &graph_data;
    table labels_data;
};

} // namespace oneapi::dal::preview::louvain::detail

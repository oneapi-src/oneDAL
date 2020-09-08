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
/// Types and descriptors of the operations for graph loading functionality

#pragma once

#include "oneapi/dal/graph/graph_common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"

namespace oneapi::dal::preview::load_graph {

/// A structure, which defines the parameters of the graph loading operation
///
/// @tparam Input  Type of the source data
/// @tparam Output Type of the destination data
template <typename Input = edge_list<int32_t>, typename Output = undirected_adjacency_array_graph<>>
struct descriptor {
    using input_type = Input;
    using output_type = Output;
};

/// Type of the descriptor output format
/// @tparam Descriptor  Type of the descriptor
template <typename Descriptor>
using output_type = typename Descriptor::output_type;

} // namespace oneapi::dal::preview::load_graph

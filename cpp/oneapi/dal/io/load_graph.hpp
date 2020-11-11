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
/// Contains the definition of the graph loading functionality

#pragma once

#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/detail/load_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph_descriptor.hpp"

namespace oneapi::dal::preview::load_graph {

/// Returns the graph object filled by data from the data source specified in the
/// descriptor
///
/// @tparam Descriptor Type of the operation descriptor
/// @tparam DataSource Type of the data source
/// @param [in] desc   The descriptor of the operation
/// @param [in] data_source The data source
///
/// @return The graph object filled by data from the data_source
template <typename Descriptor = descriptor<>, typename DataSource = graph_csv_data_source>
ONEDAL_EXPORT output_type<Descriptor> load(const Descriptor &desc, const DataSource &data_source) {
    return detail::load_impl(desc, data_source);
}
} // namespace oneapi::dal::preview::load_graph

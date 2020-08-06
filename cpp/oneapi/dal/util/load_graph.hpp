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

//#include "oneapi/dal/data/graph_service.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"

namespace oneapi::dal::preview {

struct edge_list_to_csr_descriptor {};

namespace detail {
template <typename IndexT>
edge_list<IndexT> load(const std::string &name);
}

template <typename IndexType = std::int64_t>
using edge_list = detail::graph_container<std::pair<IndexType, IndexType>>;

template <typename G, typename Descriptor, typename DataSource>
G load_graph(const Descriptor &d, const DataSource &ds) {
    G graph_data;
    convert_to_csr_impl(detail::load<typename G::vertex_type>(ds.get_filename()), graph_data);
    return graph_data;
}
} // namespace oneapi::dal::preview
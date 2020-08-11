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

#include "oneapi/dal/util/load_graph.hpp"

namespace oneapi::dal::preview::load_graph {

namespace detail {
edge_list<std::int32_t> load_edge_list(const std::string &name) {
    using int_t = std::int32_t;
    edge_list<int_t> elist;
    std::ifstream file(name);
    int_t source_vertex      = 0;
    int_t destination_vertex = 0;
    while (file >> source_vertex >> destination_vertex) {
        auto edge = std::make_pair(source_vertex, destination_vertex);
        elist.push_back(edge);
    }

    file.close();
    return elist;
}
} // namespace detail

template struct ONEAPI_DAL_EXPORT
    descriptor<edge_list<int32_t>, undirected_adjacency_array_graph<>>;

template <>
output_type<descriptor<>> load(const descriptor<> &d, const csv_data_source &ds) {
    output_type<descriptor<>> graph_data;
    convert_to_csr_impl(detail::load_edge_list(ds.get_filename()), graph_data);
    return graph_data;
}
} // namespace oneapi::dal::preview::load_graph

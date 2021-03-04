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

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"

#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include <iostream>

namespace oneapi::dal::csv::backend {

template <typename Object>
struct read_kernel_cpu {
    Object operator()(const dal::backend::context_cpu& ctx,
                      const detail::data_source_base& ds,
                      const read_args<Object>& args) const;
};

template <typename Graph>
Graph read_kernel_cpu<Graph>::operator()(const dal::backend::context_cpu& ctx,
                                         const detail::data_source_base& ds_,
                                         const read_args<Graph>& args) const {
    std::cout << "GRAPHGRAPHGRAPH EDGE LIST HERE" << std::endl;
    namespace dal = oneapi::dal;
    const dal::preview::graph_csv_data_source ds(ds_.get_file_name());
    const dal::preview::load_graph::descriptor<> d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);
    return my_graph;
}

template <>
table read_kernel_cpu<table>::operator()(const dal::backend::context_cpu& ctx,
                                         const detail::data_source_base& ds_,
                                         const read_args<table>& args) const;

} // namespace oneapi::dal::csv::backend

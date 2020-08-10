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

#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"

namespace oneapi::dal::preview::load_graph {

template <typename Input = edge_list<int32_t>, typename Output = undirected_adjacency_array_graph<>>
struct descriptor {
    using input_type  = Input;
    using output_type = Output;
};

template <typename Descriptor>
using output_type = typename Descriptor::output_type;

template <typename Descriptor = load_graph::descriptor<>, typename DataSource = csv_data_source>
output_type<Descriptor> load(const Descriptor &desc, const DataSource &data_source);
} // namespace oneapi::dal::preview::load_graph

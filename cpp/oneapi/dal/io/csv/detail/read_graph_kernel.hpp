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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include <iostream>

namespace oneapi::dal::csv::detail {

template <typename Allocator, typename Graph>
inline void read_graph_default_kernel(const dal::detail::host_policy& ctx,
                                      const detail::data_source_base& ds,
                                      const Allocator& alloc,
                                      const Graph& g) {
    std::cout << "GRAPH KERNEL" << std::endl;
    return;
}
} // namespace oneapi::dal::csv::detail

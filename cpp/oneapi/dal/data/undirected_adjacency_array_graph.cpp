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

#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"

namespace oneapi::dal::preview {

template class ONEAPI_DAL_EXPORT undirected_adjacency_array_graph<empty_value,
                                                                  empty_value,
                                                                  empty_value,
                                                                  std::int32_t,
                                                                  std::allocator<char>>;

using graph_default = undirected_adjacency_array_graph<empty_value,
                                                       empty_value,
                                                       empty_value,
                                                       std::int32_t,
                                                       std::allocator<char>>;

namespace detail {
template ONEAPI_DAL_EXPORT auto get_vertex_count_impl<graph_default>(
    const graph_default &g) noexcept -> vertex_size_type<graph_default>;

} // namespace detail

} // namespace oneapi::dal::preview

/* file: graph_common.hpp */
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

/*
//++
//  Graph types and service functionality
//--
*/

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/detail/graph_container.hpp"
#include "oneapi/dal/detail/common.hpp"

/**
 * \brief Contains graph functionality preview as an experimental part of oneapi dal.
 */
namespace oneapi::dal::preview {

template <typename G>
using vertex_user_value_type = typename G::vertex_user_value_type;

template <typename G>
using edge_user_value_type = typename G::edge_user_value_type;

template <typename G>
using vertex_type = typename G::vertex_type;

template <typename G>
using vertex_size_type = typename G::vertex_size_type;

template <typename G>
using edge_size_type = typename G::edge_size_type;

template <typename G>
using vertex_edge_size_type = typename G::vertex_edge_size_type;

template <typename G>
using edge_iterator_type = typename G::edge_iterator;

template <typename G>
using const_edge_iterator_type = typename G::const_edge_iterator;

template <typename G>
using vertex_edge_iterator_type = typename G::vertex_edge_iterator;

template <typename G>
using const_vertex_edge_iterator_type = typename G::const_vertex_edge_iterator;

template <typename G>
using edge_range_type = typename G::edge_range;

template <typename G>
using const_edge_range_type = typename G::const_edge_range;

template <typename G>
using vertex_edge_range_type = typename G::vertex_edge_range;

template <typename G>
using const_vertex_edge_range_type = typename G::const_vertex_edge_range;

template <typename IndexType = std::int32_t>
using edge_list = detail::graph_container<std::pair<IndexType, IndexType>>;

template <typename G>
using graph_allocator = typename G::allocator_type;

} // namespace oneapi::dal::preview

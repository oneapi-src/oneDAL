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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::preview {
namespace jaccard {

namespace detail {
template <typename Graph>
class vertex_similarity_input_impl;
class vertex_similarity_result_impl;
} // namespace detail

template <typename Graph>
class ONEAPI_DAL_EXPORT vertex_similarity_input {
public:
    vertex_similarity_input(const Graph& g);
    const Graph& get_graph() const;

private:
    dal::detail::pimpl<detail::vertex_similarity_input_impl<Graph>> impl_;
};

class ONEAPI_DAL_EXPORT vertex_similarity_result {
public:
    vertex_similarity_result(){};
    vertex_similarity_result(const table& vertex_pairs, const table& coeffs);
    table get_coeffs() const;
    table get_vertex_pairs() const;

private:
    dal::detail::pimpl<detail::vertex_similarity_result_impl> impl_;
};
} // namespace jaccard
} // namespace oneapi::dal::preview

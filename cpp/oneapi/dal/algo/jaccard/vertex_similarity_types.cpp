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

#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
template <typename Graph>
class detail::vertex_similarity_input_impl : public base {
public:
    vertex_similarity_input_impl(const Graph& graph_data_input, caching_builder& builder_input)
            : graph_data(graph_data_input),
              builder(builder_input) {}

    const Graph& graph_data;
    caching_builder& builder;
};

using detail::vertex_similarity_input_impl;

template <typename Graph>
vertex_similarity_input<Graph>::vertex_similarity_input(const Graph& data,
                                                        caching_builder& builder_input)
        : impl_(new vertex_similarity_input_impl<Graph>(data, builder_input)) {}

template <typename Graph>
const Graph& vertex_similarity_input<Graph>::get_graph() const {
    return impl_->graph_data;
}

template <typename Graph>
caching_builder& vertex_similarity_input<Graph>::get_caching_builder() {
    return impl_->builder;
}

class detail::vertex_similarity_result_impl : public base {
public:
    vertex_similarity_result_impl(const table& vertex_pairs,
                                  const table& coeffs,
                                  std::int64_t nonzero_coeff_count)
            : coeffs(coeffs),
              vertex_pairs(vertex_pairs),
              nonzero_coeff_count(nonzero_coeff_count) {}

    table coeffs;
    table vertex_pairs;
    std::int64_t nonzero_coeff_count;
};

template class detail::vertex_similarity_input_impl<undirected_adjacency_array_graph<>>;

template class ONEAPI_DAL_EXPORT vertex_similarity_input<undirected_adjacency_array_graph<>>;

using detail::vertex_similarity_result_impl;

vertex_similarity_result::vertex_similarity_result(const table& vertex_pairs,
                                                   const table& coeffs,
                                                   std::int64_t nonzero_coeff_count)
        : impl_(new vertex_similarity_result_impl(vertex_pairs, coeffs, nonzero_coeff_count)) {}

table vertex_similarity_result::get_coeffs() const {
    return impl_->coeffs;
}

table vertex_similarity_result::get_vertex_pairs() const {
    return impl_->vertex_pairs;
}

int64_t vertex_similarity_result::get_nonzero_coeff_count() const {
    return impl_->nonzero_coeff_count;
}
} // namespace jaccard
} // namespace oneapi::dal::preview

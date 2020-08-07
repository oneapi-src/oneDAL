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
class detail::similarity_input_impl : public base {
public:
    similarity_input_impl(const Graph &graph_data_input) : graph_data(graph_data_input) {}

    const Graph &graph_data;
};

using detail::similarity_input_impl;

template <typename Graph>
similarity_input<Graph>::similarity_input(const Graph &data)
        : impl_(new similarity_input_impl<Graph>(data)) {}

template <typename Graph>
const Graph &similarity_input<Graph>::get_graph() const {
    return impl_->graph_data;
}

class detail::similarity_result_impl : public base {
public:
    similarity_result_impl(const table &coeffs, const table &vertex_pairs)
            : coeffs(coeffs),
              vertex_pairs(vertex_pairs) {}

    table coeffs;
    table vertex_pairs;
};

template class detail::similarity_input_impl<undirected_adjacency_array_graph<> &>;

template class ONEAPI_DAL_EXPORT similarity_input<undirected_adjacency_array_graph<> &>;

using detail::similarity_result_impl;

similarity_result::similarity_result(const table &coeffs, const table &vertex_pairs)
        : impl_(new similarity_result_impl(coeffs, vertex_pairs)) {}

table similarity_result::get_coeffs() const {
    return impl_->coeffs;
}

table similarity_result::get_vertex_pairs() const {
    return impl_->vertex_pairs;
}
} // namespace jaccard
} // namespace oneapi::dal::preview

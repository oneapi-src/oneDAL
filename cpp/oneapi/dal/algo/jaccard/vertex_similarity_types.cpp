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
#include <iostream>
#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
class detail::similarity_input_impl : public base {
public:
    similarity_input_impl(const graph &graph_data_input) : graph_data(graph_data_input) {}

    const graph &graph_data;
};

using detail::similarity_input_impl;

similarity_input::similarity_input(const graph &data) : impl_(new similarity_input_impl(data)) {}

const graph &similarity_input::get_graph() const {
    return impl_->graph_data;
}

class detail::similarity_result_impl : public base {
public:
    similarity_result_impl(const array_of_floats &coefficients,
                           const array_of_pairs_uint32t &vertex_pairs)
            : coefficients(coefficients),
              vertex_pairs(vertex_pairs) {}

    array_of_floats coefficients;
    array_of_pairs_uint32t vertex_pairs;
};

using detail::similarity_result_impl;

similarity_result::similarity_result(const array_of_floats &coefficients,
                                     const array_of_pairs_uint32t &vertex_pairs)
        : impl_(new similarity_result_impl(coefficients, vertex_pairs)) {}

array_of_floats similarity_result::get_jaccard_coefficients() const {
    return impl_->coefficients;
}

array_of_pairs_uint32t similarity_result::get_vertex_pairs() const {
    return impl_->vertex_pairs;
}
} // namespace jaccard
} // namespace oneapi::dal::preview

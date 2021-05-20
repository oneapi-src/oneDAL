/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {

class detail::graph_matching_result_impl : public base {
public:
    table vertex_match;
    std::int64_t match_count;
};

using detail::graph_matching_result_impl;

graph_matching_result::graph_matching_result() : impl_(new graph_matching_result_impl()) {}

graph_matching_result::graph_matching_result(const table& vertex_match, std::int64_t match_count)
        : impl_(new graph_matching_result_impl()) {
    impl_->vertex_match = vertex_match;
    impl_->match_count = match_count;
}

table graph_matching_result::get_vertex_match() const {
    return impl_->vertex_match;
}

int64_t graph_matching_result::get_match_count() const {
    return impl_->match_count;
}
} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview

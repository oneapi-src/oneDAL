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

namespace oneapi::dal::preview::subgraph_isomorphism {

class detail::graph_matching_result_impl : public base {
public:
    table vertex_match;
    std::int64_t match_count;
};

using detail::graph_matching_result_impl;

template <typename Task>
graph_matching_result<Task>::graph_matching_result() : impl_(new graph_matching_result_impl()) {}

template <typename Task>
const table& graph_matching_result<Task>::get_vertex_match_impl() const {
    return impl_->vertex_match;
}

template <typename Task>
void graph_matching_result<Task>::set_vertex_match_impl(const table& vertex_match_table) {
    impl_->vertex_match = vertex_match_table;
}

template <typename Task>
std::int64_t graph_matching_result<Task>::get_match_count_impl() const {
    return impl_->match_count;
}

template <typename Task>
void graph_matching_result<Task>::set_match_count_impl(std::int64_t match_count_value) {
    impl_->match_count = match_count_value;
}

template class ONEDAL_EXPORT graph_matching_result<task::compute>;
} // namespace oneapi::dal::preview::subgraph_isomorphism

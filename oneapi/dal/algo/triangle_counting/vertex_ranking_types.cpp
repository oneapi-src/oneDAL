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

#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"

namespace oneapi::dal::preview::triangle_counting {

class detail::vertex_ranking_result_impl : public base {
public:
    table local_ranks;
    std::int64_t global_rank;
};

using detail::vertex_ranking_result_impl;

template <typename Task>
vertex_ranking_result<Task>::vertex_ranking_result() : impl_(new vertex_ranking_result_impl()) {}

template <typename Task>
const table& vertex_ranking_result<Task>::get_ranks_impl() const {
    return impl_->local_ranks;
}

template <typename Task>
int64_t vertex_ranking_result<Task>::get_global_rank_impl() const {
    return impl_->global_rank;
}

template <typename Task>
void vertex_ranking_result<Task>::set_ranks_impl(const table& value) {
    impl_->local_ranks = value;
}

template <typename Task>
void vertex_ranking_result<Task>::set_global_rank_impl(std::int64_t value) {
    impl_->global_rank = value;
}

template class ONEDAL_EXPORT vertex_ranking_result<task::local>;
template class ONEDAL_EXPORT vertex_ranking_result<task::global>;
template class ONEDAL_EXPORT vertex_ranking_result<task::local_and_global>;

} // namespace oneapi::dal::preview::triangle_counting

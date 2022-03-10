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

#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::shortest_paths {

class detail::traverse_result_impl : public base {
public:
    table distances;
    table predecessors;
    optional_result_id optional_result;
};

using detail::traverse_result_impl;

template <typename Task>
traverse_result<Task>::traverse_result() : impl_(new traverse_result_impl()) {}

template <typename Task>
const table& traverse_result<Task>::get_distances_impl() const {
    if (!(impl_->optional_result & optional_results::distances)) {
        throw uninitialized_optional_result(
            dal::detail::error_messages::distances_are_uninitialized());
    }
    return impl_->distances;
}

template <typename Task>
const table& traverse_result<Task>::get_predecessors_impl() const {
    if (!(impl_->optional_result & optional_results::predecessors)) {
        throw uninitialized_optional_result(
            dal::detail::error_messages::predecessors_are_uninitialized());
    }
    return impl_->predecessors;
}

template <typename Task>
void traverse_result<Task>::set_distances_impl(const table& value) {
    impl_->distances = value;
    impl_->optional_result = impl_->optional_result | optional_results::distances;
}

template <typename Task>
void traverse_result<Task>::set_predecessors_impl(const table& value) {
    impl_->predecessors = value;
    impl_->optional_result = impl_->optional_result | optional_results::predecessors;
}

template class ONEDAL_EXPORT traverse_result<task::one_to_all>;

} // namespace oneapi::dal::preview::shortest_paths

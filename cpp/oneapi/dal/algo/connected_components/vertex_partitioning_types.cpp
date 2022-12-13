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

#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::connected_components {

class detail::vertex_partitioning_result_impl : public base {
public:
    table labels;
    std::int64_t component_count;
};

using detail::vertex_partitioning_result_impl;

template <typename Task>
vertex_partitioning_result<Task>::vertex_partitioning_result()
        : impl_(new vertex_partitioning_result_impl()) {}

template <typename Task>
const table& vertex_partitioning_result<Task>::get_labels_impl() const {
    return impl_->labels;
}

template <typename Task>
int64_t vertex_partitioning_result<Task>::get_component_count_impl() const {
    return impl_->component_count;
}

template <typename Task>
void vertex_partitioning_result<Task>::set_labels_impl(const table& value) {
    impl_->labels = value;
}

template <typename Task>
void vertex_partitioning_result<Task>::set_component_count_impl(std::int64_t value) {
    impl_->component_count = value;
}

template class ONEDAL_EXPORT vertex_partitioning_result<task::vertex_partitioning>;

} // namespace oneapi::dal::preview::connected_components

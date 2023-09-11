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

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

optional_result_id get_predecessors_id() {
    return optional_result_id::get_result_id_by_index(0);
}

optional_result_id get_distances_id() {
    return optional_result_id::get_result_id_by_index(1);
}

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() {
        if constexpr (!std::is_same_v<Task, task::one_to_all>) {
            static_assert("Unsupported task");
        }
    }

    std::int64_t _source = 0;
    double _delta = 1;
    optional_result_id optional_results = optional_results::distances;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_source() const {
    return impl_->_source;
}

template <typename Task>
double descriptor_base<Task>::get_delta() const {
    return impl_->_delta;
}

template <typename Task>
void descriptor_base<Task>::set_source(std::int64_t source) {
    impl_->_source = source;
}

template <typename Task>
void descriptor_base<Task>::set_delta(double delta) {
    impl_->_delta = delta;
}

template <typename Task>
optional_result_id& descriptor_base<Task>::get_optional_results() const {
    return impl_->optional_results;
}

template <typename Task>
void descriptor_base<Task>::set_optional_results(const optional_result_id& value) {
    impl_->optional_results = value;
}

template class ONEDAL_EXPORT descriptor_base<task::one_to_all>;

} // namespace oneapi::dal::preview::shortest_paths::detail

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

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::louvain::detail {

template <typename Task>
class descriptor_impl : public base {
public:
    double accuracy_threshold = 0.0001;
    double resolution = 1.0;
    std::int64_t max_iteration_count = 10;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

template <typename Task>
double descriptor_base<Task>::get_resolution() const {
    return impl_->resolution;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

template <typename Task>
void descriptor_base<Task>::set_accuracy_threshold(double accuracy_threshold) {
    impl_->accuracy_threshold = accuracy_threshold;
}

template <typename Task>
void descriptor_base<Task>::set_resolution(double resolution) {
    impl_->resolution = resolution;
}

template <typename Task>
void descriptor_base<Task>::set_max_iteration_count(std::int64_t max_iteration_count) {
    impl_->max_iteration_count = max_iteration_count;
}

template class ONEDAL_EXPORT descriptor_base<task::vertex_partitioning>;

} // namespace oneapi::dal::preview::louvain::detail

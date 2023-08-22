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

#include "oneapi/dal/algo/rbf_kernel/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::rbf_kernel {

template <typename Task>
class detail::compute_input_impl : public base {
public:
    compute_input_impl(const table& x, const table& y) : x(x), y(y) {}
    table x;
    table y;
};

template <typename Task>
class detail::compute_result_impl : public base {
public:
    table values;
};

using detail::compute_input_impl;
using detail::compute_result_impl;

template <typename Task>
compute_input<Task>::compute_input(const table& x, const table& y)
        : impl_(new compute_input_impl<Task>(x, y)) {}

template <typename Task>
const table& compute_input<Task>::get_x() const {
    return impl_->x;
}

template <typename Task>
const table& compute_input<Task>::get_y() const {
    return impl_->y;
}

template <typename Task>
void compute_input<Task>::set_x_impl(const table& value) {
    impl_->x = value;
}

template <typename Task>
void compute_input<Task>::set_y_impl(const table& value) {
    impl_->y = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_values() const {
    return impl_->values;
}

template <typename Task>
void compute_result<Task>::set_values_impl(const table& value) {
    impl_->values = value;
}

template class ONEDAL_EXPORT compute_result<task::compute>;

} // namespace oneapi::dal::rbf_kernel

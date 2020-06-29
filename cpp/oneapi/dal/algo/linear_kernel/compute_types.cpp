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

#include "oneapi/dal/algo/linear_kernel/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::linear_kernel {

class detail::compute_input_impl : public base {
public:
    compute_input_impl(const table& x, const table& y) : x(x), y(y) {}
    table x;
    table y;
};

class detail::compute_result_impl : public base {
public:
    table values;
};

using detail::compute_input_impl;
using detail::compute_result_impl;

compute_input::compute_input(const table& x, const table& y)
        : impl_(new compute_input_impl(x, y)) {}

table compute_input::get_x() const {
    return impl_->x;
}

table compute_input::get_y() const {
    return impl_->y;
}

void compute_input::set_x_impl(const table& value) {
    impl_->x = value;
}

void compute_input::set_y_impl(const table& value) {
    impl_->y = value;
}

compute_result::compute_result() : impl_(new compute_result_impl{}) {}

table compute_result::get_values() const {
    return impl_->values;
}

void compute_result::set_values_impl(const table& value) {
    impl_->values = value;
}

} // namespace oneapi::dal::linear_kernel

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

#include "oneapi/dal/algo/linear_kernel/common.hpp"

namespace oneapi::dal::linear_kernel {

class detail::descriptor_impl : public base {
public:
    double scale = 1.0;
    double shift = 0.0;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

double descriptor_base::get_scale() const {
    return impl_->scale;
}

double descriptor_base::get_shift() const {
    return impl_->shift;
}

void descriptor_base::set_scale_impl(double value) {
    impl_->scale = value;
}

void descriptor_base::set_shift_impl(double value) {
    impl_->shift = value;
}

} // namespace oneapi::dal::linear_kernel

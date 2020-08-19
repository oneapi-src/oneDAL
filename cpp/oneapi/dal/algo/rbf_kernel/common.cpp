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

#include "oneapi/dal/algo/rbf_kernel/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::rbf_kernel {

class detail::descriptor_impl : public base {
public:
    double sigma = 1.0;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

double descriptor_base::get_sigma() const {
    return impl_->sigma;
}

void descriptor_base::set_sigma_impl(double value) {
    if (value <= 0.0) {
        throw domain_error("sigma should be > 0.0");
    }
    impl_->sigma = value;
}

} // namespace oneapi::dal::rbf_kernel

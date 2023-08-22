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

namespace oneapi::dal::rbf_kernel::detail {

template <typename Task>
class descriptor_impl : public base {
public:
    double sigma = 1.0;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_sigma() const {
    return impl_->sigma;
}

template <typename Task>
void descriptor_base<Task>::set_sigma_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::sigma_leq_zero());
    }
    impl_->sigma = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace oneapi::dal::rbf_kernel::detail

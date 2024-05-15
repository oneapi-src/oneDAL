/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/finiteness_checker/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::finiteness_checker::detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    bool allowNaN = false;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_allow_NaN() const {
    return impl_->allowNaN;
}

template <typename Task>
void descriptor_base<Task>::set_allow_NaN(bool value) {
    impl_->allowNaN = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1
} // namespace oneapi::dal::finiteness_checker::detail

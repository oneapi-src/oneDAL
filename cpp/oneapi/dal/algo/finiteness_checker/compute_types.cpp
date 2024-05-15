/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/finiteness_checker/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::finiteness_checker {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl(const table& x) : x(x) {}
    table x;
};

using detail::v1::compute_input_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& x) : impl_(new compute_input_impl<Task>(x)) {}

template <typename Task>
const table& compute_input<Task>::get_x() const {
    return impl_->x;
}

template <typename Task>
void compute_input<Task>::set_x_impl(const table& value) {
    impl_->x = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;

} // namespace v1
} // namespace oneapi::dal::finiteness_checker

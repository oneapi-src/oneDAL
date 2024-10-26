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
    compute_input_impl(const table& data) : data(data) {}
    table data;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    bool finite;
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& data) : impl_(new compute_input_impl<Task>(data)) {}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
bool compute_result<Task>::get_finite() const {
    return impl_->finite;
}

template <typename Task>
void compute_result<Task>::set_finite_impl(const bool& value) {
    impl_->finite = value;
}

template class ONEDAL_EXPORT compute_result<task::compute>;

} // namespace v1
} // namespace oneapi::dal::finiteness_checker

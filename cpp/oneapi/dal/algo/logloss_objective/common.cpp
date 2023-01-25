/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/logloss_objective/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::logloss_objective::detail {

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl(double l1_regularization_coefficient = 0
                             double l2_regularization_coefficient = 0) : 
                             l1_regularization_coefficient(l1_regularization_coefficient),
                             l2_regularization_coefficient(l2_regularization_coefficient) {}
    double l1_regularization_coefficient = 0;
    double l2_regularization_coefficient = 0;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_l1_regularization_coefficient() const {
    return impl_->l1_regularization_coefficient;
}

template <typename Task>
double descriptor_base<Task>::get_l2_regularization_coefficient() const {
    return impl_->l2_regularization_coefficient;
}


template<typename Task>
void descriptor_base<Task>::set_l1_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::invalid_argument());
    }
    impl_->l1_regularization_coefficient = value;
}

template<typename Task>
void descriptor_base<Task>::set_l2_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::invalid_argument());
    }
    impl_->l2_regularization_coefficient = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1

} // namespace oneapi::dal::objective_function::detail

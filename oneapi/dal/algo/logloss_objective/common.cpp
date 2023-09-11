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

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl(double l1_regularization_coefficient = 0.0,
                             double l2_regularization_coefficient = 0.0,
                             bool fit_intercept = true)
            : l1_regularization_coefficient(l1_regularization_coefficient),
              l2_regularization_coefficient(l2_regularization_coefficient),
              fit_intercept(fit_intercept) {}
    double l1_regularization_coefficient;
    double l2_regularization_coefficient;
    bool fit_intercept;
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

template <typename Task>
bool descriptor_base<Task>::get_intercept_flag() const {
    return impl_->fit_intercept;
}

template <typename Task>
void descriptor_base<Task>::set_l1_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::regularization_coef_is_less_than_zero());
    }
    impl_->l1_regularization_coefficient = value;
}

template <typename Task>
void descriptor_base<Task>::set_l2_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::regularization_coef_is_less_than_zero());
    }
    impl_->l2_regularization_coefficient = value;
}

template <typename Task>
void descriptor_base<Task>::set_intercept_flag_impl(bool fit_intercept) {
    impl_->fit_intercept = fit_intercept;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace oneapi::dal::logloss_objective::detail

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

#include "oneapi/dal/algo/objective_function/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::objective_function::detail {

result_option_id get_value_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_gradient_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

result_option_id get_hessian_id() {
    return result_option_id{ result_option_id::make_by_index(2) };
}

result_option_id get_packed_gradient_id() {
    return result_option_id{ result_option_id::make_by_index(3) };
}

result_option_id get_packed_hessian_id() {
    return result_option_id{ result_option_id::make_by_index(4) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

template <>
result_option_id get_default_result_options<task::logloss>() {
    return get_packed_hessian_id();
}

template <typename Task>
class descriptor_impl : public base {
public:
    result_option_id result_options = get_default_result_options<Task>();
    double l1_coef = 0;
    double l2_coef = 0;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
result_option_id descriptor_base<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
double descriptor_base<Task>::get_l1_regularization_coefficient() const {
    return impl_->l1_coef;
}

template <typename Task>
double descriptor_base<Task>::get_l2_regularization_coefficient() const {
    return impl_->l2_coef;
}

template <typename Task>
void descriptor_base<Task>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }
    impl_->result_options = value;
}

template<typename Task>
void descriptor_base<Task>::set_l1_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::invalid_argument());
    }
    impl_->l1_coef = value;
}

template<typename Task>
void descriptor_base<Task>::set_l2_regularization_coefficient_impl(double value) {
    using msg = dal::detail::error_messages;
    if (value < 0) {
        throw domain_error(msg::invalid_argument());
    }
    impl_->l2_coef = value;
}

template class ONEDAL_EXPORT descriptor_base<task::logloss>;

} // namespace oneapi::dal::objective_function::detail

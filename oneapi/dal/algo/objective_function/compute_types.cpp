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

#include "oneapi/dal/algo/objective_function/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::objective_function {

template <typename Task>
class detail::compute_input_impl : public base {
public:
    compute_input_impl(const table& data, const table& parameters, const table& responses)
            : data(data),
              parameters(parameters),
              responses(responses) {}
    table data;
    table parameters;
    table responses;
};

template <typename Task>
class detail::compute_result_impl : public base {
public:
    table value;
    table gradient;
    table hessian;
    // table packed_gradient;
    // table packed_hessian;
    result_option_id options;
};

using detail::compute_input_impl;
using detail::compute_result_impl;

template <typename Task>
compute_input<Task>::compute_input(const table& data,
                                   const table& parameters,
                                   const table& responses)
        : impl_(new compute_input_impl<Task>(data, parameters, responses)) {}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& compute_input<Task>::get_parameters() const {
    return impl_->parameters;
}

template <typename Task>
const table& compute_input<Task>::get_responses() const {
    return impl_->responses;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void compute_input<Task>::set_parameters_impl(const table& value) {
    impl_->parameters = value;
}

template <typename Task>
void compute_input<Task>::set_responses_impl(const table& value) {
    impl_->responses = value;
}

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_value() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::value)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->value;
}

template <typename Task>
const table& compute_result<Task>::get_gradient() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::gradient)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->gradient;
}

template <typename Task>
const table& compute_result<Task>::get_hessian() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::hessian)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->hessian;
}

/*
TODO add packed_gradient and packed_hessian options
*/

template <typename Task>
void compute_result<Task>::set_value_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::value)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->value = value;
}

template <typename Task>
void compute_result<Task>::set_gradient_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::gradient)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->gradient = value;
}

template <typename Task>
void compute_result<Task>::set_hessian_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::hessian)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->hessian = value;
}

/*
TODO add packed_gradient and packed_hessian options
*/

template <typename Task>
const result_option_id& compute_result<Task>::get_result_options() const {
    return impl_->options;
}

template <typename Task>
void compute_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->options = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;
template class ONEDAL_EXPORT compute_result<task::compute>;

} // namespace oneapi::dal::objective_function

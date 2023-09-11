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

/*

TODO add packed_gradient and packed_hessian options

*/

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

template <>
result_option_id get_default_result_options<task::compute>() {
    return get_hessian_id();
}

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl(const detail::objective_ptr& obj) : objective(obj) {}

    result_option_id result_options = get_default_result_options<Task>();
    detail::objective_ptr objective;
};

template <typename Task>
descriptor_base<Task>::descriptor_base()
        : impl_(new descriptor_impl<Task>{ std::make_shared<
              detail::objective<oneapi::dal::logloss_objective::descriptor<float_t>>>(
              oneapi::dal::logloss_objective::descriptor<float_t>(0, 0)) }) {}

template <typename Task>
descriptor_base<Task>::descriptor_base(const detail::objective_ptr& objective)
        : impl_(new descriptor_impl<Task>{ objective }) {}

template <typename Task>
const detail::objective_ptr& descriptor_base<Task>::get_objective_impl() const {
    return impl_->objective;
}

template <typename Task>
void descriptor_base<Task>::set_objective_impl(const detail::objective_ptr& objective) {
    impl_->objective = objective;
}

template <typename Task>
result_option_id descriptor_base<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
void descriptor_base<Task>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }
    impl_->result_options = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace oneapi::dal::objective_function::detail

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
#include "oneapi/dal/algo/logloss_objective/common.hpp"
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

namespace v1 {

template <typename Task, typename Objective>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() : desc_(new Objective()) {}

    result_option_id result_options = get_default_result_options<Task>();
    dal::detail::pimpl<Objective> desc_;
};

template <typename Task, typename Objective>
descriptor_base<Task, Objective>::descriptor_base() : impl_(new descriptor_impl<Task, Objective>{}) {}

template <typename Task, typename Objective>
result_option_id descriptor_base<Task, Objective>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task, typename Objective>
void descriptor_base<Task, Objective>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }
    impl_->result_options = value;
}


template<typename Task, typename Objective>
const auto descriptor_base<Task, Objective>::get_descriptor() const {
    return impl_->desc_;
}

template<typename Task, typename Objective>
void descriptor_base<Task, Objective>::set_descriptor_impl(const objective_t& descriptor) const {
    return impl_->desc_ = std::make_shared<Objective>(descriptor);
}


template class ONEDAL_EXPORT descriptor_base<task::logloss, logloss_objective::descriptor<float>>;
template class ONEDAL_EXPORT descriptor_base<task::logloss, logloss_objective::descriptor<double>>;

} // namespace v1

} // namespace oneapi::dal::objective_function::detail

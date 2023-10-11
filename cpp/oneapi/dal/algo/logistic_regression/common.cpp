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

#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::logistic_regression {

namespace detail {

result_option_id get_intercept_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_coefficients_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

namespace v1 {
template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() = default;

    double l1_coef = 0.0;
    double l2_coef = 0.0;
    bool compute_intercept = true;
    std::int32_t max_iter = 100;
    double tol = 1e-4;
    std::int64_t class_count = 2;
    optimizer_enum opt;

    result_option_id result_options = get_default_result_options<Task>();
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

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
    else if (!get_compute_intercept() && value.test(result_options::intercept)) {
        throw domain_error(msg::intercept_result_option_requires_intercept_flag());
    }
    impl_->result_options = value;
}

template <typename Task>
descriptor_base<Task>::descriptor_base(bool compute_intercept,
                                       double l2_coef,
                                       std::int32_t max_iter,
                                       double tol,
                                       optimizer_enum opt)
        : impl_(new descriptor_impl<Task>{}) {
    impl_->compute_intercept = compute_intercept;
    impl_->l2_coef = l2_coef;
    impl_->tol = tol;
    impl_->max_iter = max_iter;
    impl_->opt = opt;
}

template <typename Task>
bool descriptor_base<Task>::get_compute_intercept() const {
    return impl_->compute_intercept;
}

// template <typename Task>
// double descriptor_base<Task>::get_l1_coef() const {
//     return impl_->l1_coef;
// }

template <typename Task>
double descriptor_base<Task>::get_l2_coef() const {
    return impl_->l2_coef;
}

template <typename Task>
double descriptor_base<Task>::get_tol() const {
    return impl_->tol;
}

template <typename Task>
std::int32_t descriptor_base<Task>::get_max_iter() const {
    return impl_->max_iter;
}

// template <typename Task>
// std::int64_t descriptor_base<Task>::get_class_count() const {
//     return impl_->class_count;
// }

template <typename Task>
optimizer_enum descriptor_base<Task>::get_optimizer() const {
    return impl_->opt;
}

template <typename Task>
void descriptor_base<Task>::set_compute_intercept_impl(bool compute_intercept) {
    impl_->compute_intercept = compute_intercept;
}

// template <typename Task>
// void descriptor_base<Task>::set_l1_coef_impl(double l1_coef) {
//     impl_->l1_coef = l1_coef;
// }

template <typename Task>
void descriptor_base<Task>::set_l2_coef_impl(double l2_coef) {
    impl_->l2_coef = l2_coef;
}

template <typename Task>
void descriptor_base<Task>::set_tol_impl(double tol) {
    impl_->tol = tol;
}

template <typename Task>
void descriptor_base<Task>::set_max_iter_impl(std::int32_t max_iter) {
    impl_->max_iter = max_iter;
}

template <typename Task>
void descriptor_base<Task>::set_optimizer_impl(optimizer_enum opt) {
    impl_->opt = opt;
}

// template <typename Task>
// void descriptor_base<Task>::set_class_count_impl(std::int64_t class_count) {
//     impl_->class_count = class_count;
// }

template class ONEDAL_EXPORT descriptor_base<task::binary_classification>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_{ std::make_shared<detail::model_impl<Task>>() } {}

template <typename Task>
model<Task>::model(const std::shared_ptr<detail::model_impl<Task>>& impl) : impl_{ impl } {}

template <typename Task>
const table& model<Task>::get_packed_coefficients() const {
    return impl_->get_packed_coefficients();
}

template <typename Task>
model<Task>& model<Task>::set_packed_coefficients(const table& t) {
    impl_->set_packed_coefficients(t);
    return *this;
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::binary_classification>;

// ONEDAL_REGISTER_SERIALIZABLE(detail::model_impl<task::binary_classification>)

} // namespace v1
} // namespace oneapi::dal::logistic_regression

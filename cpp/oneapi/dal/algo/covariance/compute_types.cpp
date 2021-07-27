/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::covariance {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl(const table& input) : input(input) {}
    table input;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    table cov_matrix;
    table cor_matrix;
    table means;
    result_option_id options = get_default_result_options<Task>();
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& x)
        : impl_(new compute_input_impl<Task>(x)) {}

template <typename Task>
const table& compute_input<Task>::get_input() const {
    return impl_->input;
}

template <typename Task>
void compute_input<Task>::set_input_impl(const table& value) {
    impl_->x = value;
}

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_cov_matrix() const {
    using msg = dal::detail::error_messages;
    if (!bool(get_result_options() & result_options::cov_matrix)) {
        throw domain_error(msg::result_option_have_not_been_computed());
    }
    return impl_->cov_matrix;
}

template <typename Task>
void compute_result<Task>::set_cov_matrix_impl(const table& value) {
    impl_->cov_matrix = value;
}
template <typename Task>
const table& compute_result<Task>::get_cor_matrix() const {
    using msg = dal::detail::error_messages;
    if (!bool(get_result_options() & result_options::cor_matrix)) {
        throw domain_error(msg::result_option_have_not_been_computed());
    }
    return impl_->cor_matrix;
}

template <typename Task>
void compute_result<Task>::set_cor_matrix_impl(const table& value) {
    impl_->cor_matrix = value;
}

template <typename Task>
const table& compute_result<Task>::get_means() const {
    using msg = dal::detail::error_messages;
    if (!bool(get_result_options() & result_options::means)) {
        throw domain_error(msg::result_option_have_not_been_computed());
    }
    return impl_->means;
}

template <typename Task>
void compute_result<Task>::set_means_impl(const table& value) {
    impl_->means = value;
}

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

} // namespace v1
} // namespace oneapi::dal::covariance

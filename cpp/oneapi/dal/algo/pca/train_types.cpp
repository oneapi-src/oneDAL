/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/pca/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::pca {

template <typename Task>
class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data) : data(data) {}
    table data;
};

template <typename Task>
class detail::train_result_impl : public base {
public:
    model<Task> trained_model;
    table eigenvalues;
    table variances;
    table means;
    result_option_id result_options;
};

using detail::train_input_impl;
using detail::train_result_impl;

template <typename Task>
train_input<Task>::train_input(const table& data) : impl_(new train_input_impl<Task>(data)) {}

template <typename Task>
const table& train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void train_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

using msg = dal::detail::error_messages;

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& train_result<Task>::get_eigenvalues() const {
    if (!get_result_options().test(result_options::eigenvalues)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->eigenvalues;
}

template <typename Task>
const table& train_result<Task>::get_eigenvectors() const {
    if (!get_result_options().test(result_options::eigenvectors)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->trained_model.get_eigenvectors();
}

template <typename Task>
const table& train_result<Task>::get_variances() const {
    if (!get_result_options().test(result_options::vars)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->variances;
}

template <typename Task>
const table& train_result<Task>::get_means() const {
    if (!get_result_options().test(result_options::means)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->means;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_eigenvalues_impl(const table& value) {
    if (!get_result_options().test(result_options::eigenvalues)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->eigenvalues = value;
}

template <typename Task>
void train_result<Task>::set_variances_impl(const table& value) {
    if (!get_result_options().test(result_options::vars)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->variances = value;
}

template <typename Task>
void train_result<Task>::set_means_impl(const table& value) {
    if (!get_result_options().test(result_options::means)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->means = value;
}

template <typename Task>
const result_option_id& train_result<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
void train_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->result_options = value;
}

template class ONEDAL_EXPORT train_input<task::dim_reduction>;
template class ONEDAL_EXPORT train_result<task::dim_reduction>;

} // namespace oneapi::dal::pca

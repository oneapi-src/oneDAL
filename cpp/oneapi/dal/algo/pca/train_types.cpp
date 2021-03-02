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

#include "oneapi/dal/algo/pca/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::pca {

template <typename Task>
class detail::v1::train_input_impl : public base {
public:
    train_input_impl(const table& data) : data(data) {}

    table data;
};

template <typename Task>
class detail::v1::train_result_impl : public base {
public:
    model<Task> trained_model;
    table eigenvalues;
    table variances;
    table means;
};

using detail::v1::train_input_impl;
using detail::v1::train_result_impl;

namespace v1 {

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

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& train_result<Task>::get_eigenvalues() const {
    return impl_->eigenvalues;
}

template <typename Task>
const table& train_result<Task>::get_eigenvectors() const {
    return impl_->trained_model.get_eigenvectors();
}

template <typename Task>
const table& train_result<Task>::get_variances() const {
    return impl_->variances;
}

template <typename Task>
const table& train_result<Task>::get_means() const {
    return impl_->means;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_eigenvalues_impl(const table& value) {
    impl_->eigenvalues = value;
}

template <typename Task>
void train_result<Task>::set_variances_impl(const table& value) {
    impl_->variances = value;
}

template <typename Task>
void train_result<Task>::set_means_impl(const table& value) {
    impl_->means = value;
}

template class ONEDAL_EXPORT train_input<task::dim_reduction>;
template class ONEDAL_EXPORT train_result<task::dim_reduction>;

} // namespace v1
} // namespace oneapi::dal::pca

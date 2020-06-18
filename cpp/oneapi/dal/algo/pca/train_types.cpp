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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/pca/train_types.hpp"

namespace dal {
namespace decomposition {
namespace pca {

class detail::train_input_impl : public base {
  public:
    train_input_impl(const table& data)
        : data(data) {}

    table data;
};

class detail::train_result_impl : public base {
  public:
    model trained_model;
    table eigenvalues;
    table explained_variance;
};

using detail::train_input_impl;
using detail::train_result_impl;

train_input::train_input(const table& data)
    : impl_(new train_input_impl(data)) {}

table train_input::get_data() const {
    return impl_->data;
}

void train_input::set_data_impl(const table& value) {
    impl_->data = value;
}

train_result::train_result() : impl_(new train_result_impl{}) {}

model train_result::get_model() const {
    return impl_->trained_model;
}

table train_result::get_eigenvalues() const {
    return impl_->eigenvalues;
}

table train_result::get_eigenvectors() const {
    return impl_->trained_model.get_eigenvectors();
}

table train_result::get_explained_variance() const {
    return impl_->explained_variance;
}

void train_result::set_model_impl(const model& value) {
    impl_->trained_model = value;
}

void train_result::set_eigenvalues_impl(const table& value) {
    impl_->eigenvalues = value;
}

void train_result::set_explained_variance_impl(const table& value) {
    impl_->explained_variance = value;
}

} // namespace pca
} // namespace decomposition
} // namespace dal

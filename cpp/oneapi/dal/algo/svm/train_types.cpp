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

#include "oneapi/dal/algo/svm/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::svm {

class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& labels, const table& weights)
            : data(data),
              labels(labels),
              weights(weights) {}
    table data;
    table labels;
    table weights;
};

class detail::train_result_impl : public base {
public:
    model trained_model;
    table support_indices;
};

using detail::train_input_impl;
using detail::train_result_impl;

train_input::train_input(const table& data, const table& labels, const table& weights)
        : impl_(new train_input_impl(data, labels, weights)) {}

table train_input::get_data() const {
    return impl_->data;
}

table train_input::get_labels() const {
    return impl_->labels;
}

table train_input::get_weights() const {
    return impl_->weights;
}

void train_input::set_data_impl(const table& value) {
    impl_->data = value;
}

void train_input::set_labels_impl(const table& value) {
    impl_->labels = value;
}

void train_input::set_weights_impl(const table& value) {
    impl_->weights = value;
}

train_result::train_result() : impl_(new train_result_impl{}) {}

model train_result::get_model() const {
    return impl_->trained_model;
}

table train_result::get_support_vectors() const {
    return impl_->trained_model.get_support_vectors();
}

table train_result::get_support_indices() const {
    return impl_->support_indices;
}

table train_result::get_coeffs() const {
    return impl_->trained_model.get_coeffs();
}

double train_result::get_bias() const {
    return impl_->trained_model.get_bias();
}

std::int64_t train_result::get_support_vector_count() const {
    return impl_->trained_model.get_support_vector_count();
}

void train_result::set_model_impl(const model& value) {
    impl_->trained_model = value;
}

void train_result::set_support_vectors_impl(const table& value) {
    impl_->trained_model.set_support_vectors(value);
}
void train_result::set_support_indices_impl(const table& value) {
    impl_->support_indices = value;
}

void train_result::set_coeffs_impl(const table& value) {
    impl_->trained_model.set_coeffs(value);
}

void train_result::set_bias_impl(double value) {
    impl_->trained_model.set_bias(value);
}

void train_result::set_support_vector_count_impl(std::int64_t value) {
    impl_->trained_model.set_support_vector_count(value);
}

} // namespace oneapi::dal::svm

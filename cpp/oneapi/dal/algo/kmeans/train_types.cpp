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

#include "oneapi/dal/algo/kmeans/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::kmeans {

class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data) : data(data) {}
    train_input_impl(const table& data, const table& initial_centroids)
            : data(data),
              initial_centroids(initial_centroids) {}

    table data;
    table initial_centroids;
};

class detail::train_result_impl : public base {
public:
    model trained_model;
    table labels;
    std::int64_t iteration_count;
    double objective_function_value;
};

using detail::train_input_impl;
using detail::train_result_impl;

train_input::train_input(const table& data) : impl_(new train_input_impl(data)) {}

train_input::train_input(const table& data, const table& initial_centroids)
        : impl_(new train_input_impl(data, initial_centroids)) {}

table train_input::get_data() const {
    return impl_->data;
}

table train_input::get_initial_centroids() const {
    return impl_->initial_centroids;
}

void train_input::set_data_impl(const table& value) {
    impl_->data = value;
}

void train_input::set_initial_centroids_impl(const table& value) {
    impl_->initial_centroids = value;
}

train_result::train_result() : impl_(new train_result_impl{}) {}

model train_result::get_model() const {
    return impl_->trained_model;
}

table train_result::get_labels() const {
    return impl_->labels;
}

std::int64_t train_result::get_iteration_count() const {
    return impl_->iteration_count;
}

double train_result::get_objective_function_value() const {
    return impl_->objective_function_value;
}

void train_result::set_model_impl(const model& value) {
    impl_->trained_model = value;
}

void train_result::set_labels_impl(const table& value) {
    impl_->labels = value;
}

void train_result::set_iteration_count_impl(std::int64_t value) {
    impl_->iteration_count = value;
}

void train_result::set_objective_function_value_impl(double value) {
    impl_->objective_function_value = value;
}

} // namespace oneapi::dal::kmeans

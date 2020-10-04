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
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/network/network.hpp"

namespace oneapi::dal::kmeans {

template <typename Task>
class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data) : data(data), network(net) {}
    train_input_impl(const table& data, const table& initial_centroids)
            : data(data),
              initial_centroids(initial_centroids), network(net) {}

    train_input_impl(const table& data, const table& initial_centroids, const oneapi::dal::network::network& network)
            : data(data),
              initial_centroids(initial_centroids), network(network) {}

    table data;
    table initial_centroids;
    oneapi::dal::network::empty_network net;
    const oneapi::dal::network::network& network;
};

template <typename Task>
class detail::train_result_impl : public base {
public:
    model<Task> trained_model;
    table labels;
    std::int64_t iteration_count = 0;
    double objective_function_value = 0.0;
};

using detail::train_input_impl;
using detail::train_result_impl;

template <typename Task>
train_input<Task>::train_input(const table& data) : impl_(new train_input_impl(data)) {}

template <typename Task>
train_input<Task>::train_input(const table& data, const table& initial_centroids)
        : impl_(new train_input_impl<Task>(data, initial_centroids)) {}

template <typename Task>
train_input<Task>::train_input(const table& data, const table& initial_centroids, const oneapi::dal::network::network& network)
        : impl_(new train_input_impl<Task>(data, initial_centroids, network)) {}

template <typename Task>
table train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
table train_input<Task>::get_initial_centroids() const {
    return impl_->initial_centroids;
}

template <typename Task>
const oneapi::dal::network::network& train_input<Task>::get_network() const {
    return impl_->network;
}

template <typename Task>
void train_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void train_input<Task>::set_initial_centroids_impl(const table& value) {
    impl_->initial_centroids = value;
}

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl{}) {}

template <typename Task>
model<Task> train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
table train_result<Task>::get_labels() const {
    return impl_->labels;
}

template <typename Task>
std::int64_t train_result<Task>::get_iteration_count() const {
    return impl_->iteration_count;
}

template <typename Task>
double train_result<Task>::get_objective_function_value() const {
    return impl_->objective_function_value;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_labels_impl(const table& value) {
    impl_->labels = value;
}

template <typename Task>
void train_result<Task>::set_iteration_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error("iteration_count should be >= 0");
    }
    impl_->iteration_count = value;
}

template <typename Task>
void train_result<Task>::set_objective_function_value_impl(double value) {
    if (value < 0.0) {
        throw domain_error("objective_function_value should be >= 0");
    }
    impl_->objective_function_value = value;
}

template class ONEAPI_DAL_EXPORT train_input<task::clustering>;
template class ONEAPI_DAL_EXPORT train_result<task::clustering>;

} // namespace oneapi::dal::kmeans

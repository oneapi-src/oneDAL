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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

template <typename Task>
class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& labels, std::uint64_t results_to_compute = 0)
            : data(data),
              labels(labels),
              results_to_compute(results_to_compute) {}
    train_input_impl(const table& data,
                     const table& labels,
                     train_result_to_compute results_to_compute)
            : data(data),
              labels(labels),
              results_to_compute(static_cast<std::uint64_t>(results_to_compute)) {}

    table data;
    table labels;
    std::uint64_t results_to_compute = 0;
};

template <typename Task>
class detail::train_result_impl : public base {
public:
    model<Task> trained_model;

    table oob_err;
    table oob_per_observation_err;
    table variable_importance;
};

using detail::train_input_impl;
using detail::train_result_impl;

template <typename Task>
train_input<Task>::train_input(const table& data,
                               const table& labels,
                               std::uint64_t results_to_compute)
        : impl_(new train_input_impl<Task>(data, labels, results_to_compute)) {}

template <typename Task>
train_input<Task>::train_input(const table& data,
                               const table& labels,
                               train_result_to_compute results_to_compute)
        : impl_(new train_input_impl<Task>(data, labels, results_to_compute)) {}

template <typename Task>
table train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
table train_input<Task>::get_labels() const {
    return impl_->labels;
}

template <typename Task>
std::uint64_t train_input<Task>::get_results_to_compute() const {
    return impl_->results_to_compute;
}

template <typename Task>
void train_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void train_input<Task>::set_labels_impl(const table& value) {
    impl_->labels = value;
}

template <typename Task>
void train_input<Task>::set_results_to_compute_impl(std::uint64_t value) {
    impl_->results_to_compute = value;
}

template class ONEAPI_DAL_EXPORT train_input<task::classification>;
template class ONEAPI_DAL_EXPORT train_input<task::regression>;

/* train_result implementation*/
template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
model<Task> train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
table train_result<Task>::get_oob_err() const {
    return impl_->oob_err;
}

template <typename Task>
table train_result<Task>::get_oob_per_observation_err() const {
    return impl_->oob_per_observation_err;
}

template <typename Task>
table train_result<Task>::get_var_importance() const {
    return impl_->variable_importance;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_impl(const table& value) {
    impl_->oob_err = value;
}

template <typename Task>
void train_result<Task>::set_oob_per_observation_err_impl(const table& value) {
    impl_->oob_per_observation_err = value;
}

template <typename Task>
void train_result<Task>::set_var_importance_impl(const table& value) {
    impl_->variable_importance = value;
}

template class ONEAPI_DAL_EXPORT train_result<task::classification>;
template class ONEAPI_DAL_EXPORT train_result<task::regression>;

} // namespace oneapi::dal::decision_forest

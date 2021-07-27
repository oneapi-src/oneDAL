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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

template <typename Task>
class detail::v1::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& responses)
            : data(data),
              responses(responses) {}

    table data;
    table responses;
};

template <typename Task>
class detail::v1::train_result_impl : public base {
public:
    model<Task> trained_model;

    table oob_err;
    table oob_err_per_observation;
    table variable_importance;
};

using detail::v1::train_input_impl;
using detail::v1::train_result_impl;

namespace v1 {

template <typename Task>
train_input<Task>::train_input(const table& data, const table& responses)
        : impl_(new train_input_impl<Task>(data, responses)) {}

template <typename Task>
const table& train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& train_input<Task>::get_responses() const {
    return impl_->responses;
}

template <typename Task>
void train_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void train_input<Task>::set_responses_impl(const table& value) {
    impl_->responses = value;
}

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& train_result<Task>::get_oob_err() const {
    return impl_->oob_err;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_per_observation() const {
    return impl_->oob_err_per_observation;
}

template <typename Task>
const table& train_result<Task>::get_var_importance() const {
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
void train_result<Task>::set_oob_err_per_observation_impl(const table& value) {
    impl_->oob_err_per_observation = value;
}

template <typename Task>
void train_result<Task>::set_var_importance_impl(const table& value) {
    impl_->variable_importance = value;
}

template class ONEDAL_EXPORT train_input<task::classification>;
template class ONEDAL_EXPORT train_input<task::regression>;
template class ONEDAL_EXPORT train_result<task::classification>;
template class ONEDAL_EXPORT train_result<task::regression>;

} // namespace v1
} // namespace oneapi::dal::decision_forest

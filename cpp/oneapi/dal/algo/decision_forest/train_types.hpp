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

#pragma once

#include "oneapi/dal/algo/decision_forest/common.hpp"

namespace oneapi::dal::decision_forest {

namespace detail {
template <typename Task = task::by_default>
class train_input_impl;

template <typename Task = task::by_default>
class train_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class train_input : public base {
public:
    using pimpl = dal::detail::pimpl<detail::train_input_impl<Task>>;
    train_input(const table& data, const table& labels, std::uint64_t results_to_compute = 0);
    train_input(const table& data, const table& labels, train_result_to_compute results_to_compute);

    table get_data() const;
    table get_labels() const;
    std::uint64_t get_results_to_compute() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }
    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    auto& set_results_to_compute(train_result_to_compute value) {
        set_results_to_compute_impl(static_cast<std::uint64_t>(value));
        return *this;
    }
    auto& set_results_to_compute(std::uint64_t value) {
        set_results_to_compute_impl(value);
        return *this;
    }

private:
    void set_data_impl(const table& value);
    void set_labels_impl(const table& value);
    void set_results_to_compute_impl(std::uint64_t value);

    pimpl impl_;
};

template <typename Task = task::by_default>
class train_result {
public:
    using pimpl = dal::detail::pimpl<detail::train_result_impl<Task>>;
    train_result();

    model<Task> get_model() const;

    table get_oob_err() const;
    table get_oob_per_observation_err() const;
    table get_var_importance() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    auto& set_oob_err(const table& value) {
        set_oob_err_impl(value);
        return *this;
    }

    auto& set_oob_per_observation_err(const table& value) {
        set_oob_per_observation_err_impl(value);
        return *this;
    }

    auto& set_var_importance(const table& value) {
        set_var_importance_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<Task>&);

    void set_oob_err_impl(const table&);
    void set_oob_per_observation_err_impl(const table&);
    void set_var_importance_impl(const table&);

    pimpl impl_;
};

} // namespace oneapi::dal::decision_forest

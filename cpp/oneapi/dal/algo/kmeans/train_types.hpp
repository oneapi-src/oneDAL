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

#include "oneapi/dal/algo/kmeans/common.hpp"
#include "oneapi/dal/network/network.hpp"

namespace oneapi::dal::kmeans {

namespace detail {
template <typename Task = task::by_default>
class train_input_impl;

template <typename Task = task::by_default>
class train_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class ONEAPI_DAL_EXPORT train_input : public base {
public:
    using task_t = Task;
    train_input(const table& data);
    train_input(const table& data, const table& initial_centroids);
    train_input(const table& data, const table& initial_centroids, const oneapi::dal::network::network& network);

    table get_data() const;
    table get_initial_centroids() const;

    const oneapi::dal::network::network& get_network() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    auto& set_initial_centroids(const table& data) {
        set_initial_centroids_impl(data);
        return *this;
    }

private:
    void set_data_impl(const table& data);
    void set_initial_centroids_impl(const table& data);

    dal::detail::pimpl<detail::train_input_impl<task_t>> impl_;
};

template <typename Task = task::by_default>
class ONEAPI_DAL_EXPORT train_result {
public:
    using task_t = Task;
    train_result();

    model<task_t> get_model() const;
    table get_labels() const;
    int64_t get_iteration_count() const;
    double get_objective_function_value() const;

    auto& set_model(const model<task_t>& value) {
        set_model_impl(value);
        return *this;
    }

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    auto& set_iteration_count(std::int64_t value) {
        set_iteration_count_impl(value);
        return *this;
    }
    auto& set_objective_function_value(double value) {
        set_objective_function_value_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<task_t>&);
    void set_labels_impl(const table&);
    void set_iteration_count_impl(std::int64_t);
    void set_objective_function_value_impl(double);

    dal::detail::pimpl<detail::train_result_impl<task_t>> impl_;
};

} // namespace oneapi::dal::kmeans

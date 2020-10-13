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

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init {

namespace detail {
template <typename Task = task::by_default>
class compute_input_impl;

template <typename Task = task::by_default>
class compute_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class ONEAPI_DAL_EXPORT compute_input : public base {
public:
    using task_t = Task;

    compute_input(const table& data);

    table get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

private:
    void set_data_impl(const table& data);

    dal::detail::pimpl<detail::compute_input_impl<task_t>> impl_;
};

template <typename Task = task::by_default>
class ONEAPI_DAL_EXPORT compute_result {
public:
    using task_t = Task;

    compute_result();

    table get_centroids() const;

    auto& set_centroids(const table& value) {
        set_centroids_impl(value);
        return *this;
    }

private:
    void set_centroids_impl(const table&);

    dal::detail::pimpl<detail::compute_result_impl<task_t>> impl_;
};

} // namespace oneapi::dal::kmeans_init

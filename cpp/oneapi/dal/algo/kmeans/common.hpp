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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::kmeans {

namespace task {
struct clustering {};
using by_default = clustering;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

template <typename Task = task::by_default>
class model_impl;
} // namespace detail

namespace method {
struct lloyd_dense {};
using by_default = lloyd_dense;
} // namespace method

template <typename Task = task::by_default>
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using task_t = Task;
    using float_t = float;
    using method_t = method::by_default;

    descriptor_base();

    std::int64_t get_cluster_count() const;
    std::int64_t get_max_iteration_count() const;
    double get_accuracy_threshold() const;

protected:
    void set_cluster_count_impl(std::int64_t);
    void set_max_iteration_count_impl(std::int64_t);
    void set_accuracy_threshold_impl(double);

    dal::detail::pimpl<detail::descriptor_impl<task_t>> impl_;
};

template <typename Float = descriptor_base<task::by_default>::float_t,
          typename Method = descriptor_base<task::by_default>::method_t,
          typename Task = task::by_default>
class descriptor : public descriptor_base<Task> {
public:
    using float_t = Float;
    using method_t = Method;

    explicit descriptor(std::int64_t cluster_count = 2) {
        set_cluster_count(cluster_count);
    }

    auto& set_cluster_count(int64_t value) {
        descriptor_base<Task>::set_cluster_count_impl(value);
        return *this;
    }

    auto& set_max_iteration_count(int64_t value) {
        descriptor_base<Task>::set_max_iteration_count_impl(value);
        return *this;
    }

    auto& set_accuracy_threshold(double value) {
        descriptor_base<Task>::set_accuracy_threshold_impl(value);
        return *this;
    }
};

template <typename Task = task::by_default>
class ONEDAL_EXPORT model : public base {
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;
    model();

    table get_centroids() const;
    std::int64_t get_cluster_count() const;

    auto& set_centroids(const table& value) {
        set_centroids_impl(value);
        return *this;
    }

private:
    void set_centroids_impl(const table&);

    dal::detail::pimpl<detail::model_impl<task_t>> impl_;
};

} // namespace oneapi::dal::kmeans

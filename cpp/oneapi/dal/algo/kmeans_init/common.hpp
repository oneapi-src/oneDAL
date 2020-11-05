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

namespace oneapi::dal::kmeans_init {

namespace task {
namespace v1 {
struct init {};
using by_default = init;
} // namespace v1

using v1::init;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
struct dense {};
struct random_dense {};
struct plus_plus_dense {};
struct parallel_plus_dense {};
using by_default = dense;
} // namespace v1

using v1::dense;
using v1::random_dense;
using v1::plus_plus_dense;
using v1::parallel_plus_dense;
using v1::by_default;

} // namespace method

namespace detail {
namespace v1 {
struct tag {};
template <typename Task>
class descriptor_impl;
} // namespace v1

using v1::tag;
using v1::descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method,
                                                            method::dense,
                                                            method::random_dense,
                                                            method::plus_plus_dense,
                                                            method::parallel_plus_dense>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::init>;

} // namespace detail

namespace v1 {

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_cluster_count() const;

protected:
    void set_cluster_count_impl(std::int64_t);

private:
    dal::detail::pimpl<detail::descriptor_impl<Task>> impl_;
};

template <typename Float = descriptor_base<>::float_t,
          typename Method = descriptor_base<>::method_t,
          typename Task = descriptor_base<>::task_t>
class descriptor : public descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    explicit descriptor(std::int64_t cluster_count = 2) {
        set_cluster_count(cluster_count);
    }

    auto& set_cluster_count(int64_t value) {
        descriptor_base<Task>::set_cluster_count_impl(value);
        return *this;
    }
};

} // namespace v1

using v1::descriptor_base;
using v1::descriptor;

} // namespace oneapi::dal::kmeans_init

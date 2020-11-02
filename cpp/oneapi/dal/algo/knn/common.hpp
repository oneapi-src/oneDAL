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

namespace oneapi::dal::knn {

namespace task {
struct classification {};
using by_default = classification;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

class model_impl;
} // namespace detail

namespace method {
struct kd_tree {};
struct brute_force {};
using by_default = brute_force;
} // namespace method

template <typename Task = task::by_default>
class descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_class_count() const;
    std::int64_t get_neighbor_count() const;

protected:
    void set_class_count_impl(std::int64_t value);
    void set_neighbor_count_impl(std::int64_t value);

    dal::detail::pimpl<detail::descriptor_impl<task_t>> impl_;
};

template <typename Float = descriptor_base<task::by_default>::float_t,
          typename Method = descriptor_base<task::by_default>::method_t,
          typename Task = task::by_default>
class descriptor : public descriptor_base<Task> {
public:
    using tag_t = detail::tag;
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    explicit descriptor(std::int64_t class_count, std::int64_t neighbor_count) {
        set_class_count(class_count);
        set_neighbor_count(neighbor_count);
    }

    auto& set_class_count(std::int64_t value) {
        descriptor_base<task_t>::set_class_count_impl(value);
        return *this;
    }

    auto& set_neighbor_count(std::int64_t value) {
        descriptor_base<task_t>::set_neighbor_count_impl(value);
        return *this;
    }
};

template <typename Task = task::by_default>
class model : public base {
    friend dal::detail::pimpl_accessor;

public:
    model();

private:
    explicit model(const std::shared_ptr<detail::model_impl>& impl);
    dal::detail::pimpl<detail::model_impl> impl_;
};

} // namespace oneapi::dal::knn

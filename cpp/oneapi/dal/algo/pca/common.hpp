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

namespace oneapi::dal::pca {

namespace task {
struct dim_reduction {};
using by_default = dim_reduction;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

template <typename Task = task::by_default>
class model_impl;
} // namespace detail

namespace method {
struct cov {};
struct svd {};
using by_default = cov;
} // namespace method

template <typename Task = task::by_default>
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using task_t = Task;
    using float_t = float;
    using method_t = method::by_default;

    descriptor_base();

    auto get_component_count() const -> std::int64_t;
    auto get_deterministic() const -> bool;

protected:
    void set_component_count_impl(std::int64_t value);
    void set_deterministic_impl(bool value);

    dal::detail::pimpl<detail::descriptor_impl<task_t>> impl_;
};

template <typename Float = descriptor_base<task::by_default>::float_t,
          typename Method = descriptor_base<task::by_default>::method_t,
          typename Task = task::by_default>
class descriptor : public descriptor_base<Task> {
public:
    using float_t = Float;
    using method_t = Method;

    explicit descriptor(std::int64_t component_count = 0) {
        set_component_count(component_count);
    }

    auto& set_component_count(int64_t value) {
        descriptor_base<Task>::set_component_count_impl(value);
        return *this;
    }

    auto& set_deterministic(bool value) {
        descriptor_base<Task>::set_deterministic_impl(value);
        return *this;
    }
};

template <typename Task = task::by_default>
class ONEDAL_EXPORT model : public base {
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;
    model();

    table get_eigenvectors() const;

    auto& set_eigenvectors(const table& value) {
        set_eigenvectors_impl(value);
        return *this;
    }

private:
    void set_eigenvectors_impl(const table&);

    dal::detail::pimpl<detail::model_impl<task_t>> impl_;
};

} // namespace oneapi::dal::pca

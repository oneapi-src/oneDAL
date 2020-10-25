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

namespace oneapi::dal::linear_kernel {

namespace task {
struct kernel_function {};
using by_default = kernel_function;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

} // namespace detail

namespace method {
struct dense {};
struct csr {};
using by_default = dense;
} // namespace method

template <typename Task = task::by_default>
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    double get_scale() const;
    double get_shift() const;

protected:
    void set_scale_impl(double value);
    void set_shift_impl(double value);

    dal::detail::pimpl<detail::descriptor_impl<task_t>> impl_;
};

template <typename Float = descriptor_base<task::by_default>::float_t,
          typename Method = descriptor_base<task::by_default>::method_t,
          typename Task = task::by_default>
class descriptor : public descriptor_base<Task> {
public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    auto& set_scale(double value) {
        descriptor_base<task_t>::set_scale_impl(value);
        return *this;
    }

    auto& set_shift(double value) {
        descriptor_base<task_t>::set_shift_impl(value);
        return *this;
    }
};

} // namespace oneapi::dal::linear_kernel

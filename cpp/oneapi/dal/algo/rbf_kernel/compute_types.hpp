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

#include "oneapi/dal/algo/rbf_kernel/common.hpp"

namespace oneapi::dal::rbf_kernel {

namespace detail {
class compute_input_impl;
class compute_result_impl;
} // namespace detail

class ONEDAL_EXPORT compute_input : public base {
public:
    compute_input(const table& x, const table& y);

    table get_x() const;
    table get_y() const;

    auto& set_x(const table& data) {
        set_x_impl(data);
        return *this;
    }

    auto& set_y(const table& data) {
        set_y_impl(data);
        return *this;
    }

private:
    void set_x_impl(const table& data);
    void set_y_impl(const table& data);

    dal::detail::pimpl<detail::compute_input_impl> impl_;
};

class ONEDAL_EXPORT compute_result : public base {
public:
    compute_result();

    table get_values() const;

    auto& set_values(const table& value) {
        set_values_impl(value);
        return *this;
    }

private:
    void set_values_impl(const table&);

    dal::detail::pimpl<detail::compute_result_impl> impl_;
};

} // namespace oneapi::dal::rbf_kernel

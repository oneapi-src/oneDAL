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

#include "oneapi/dal/algo/pca/common.hpp"

namespace oneapi::dal::pca {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_input(const table& data);

    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

protected:
    void set_data_impl(const table& data);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_result();

    const table& get_eigenvectors() const;

    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    const table& get_eigenvalues() const;

    auto& set_eigenvalues(const table& value) {
        set_eigenvalues_impl(value);
        return *this;
    }

    const table& get_variances() const;

    auto& set_variances(const table& value) {
        set_variances_impl(value);
        return *this;
    }

    const table& get_means() const;

    auto& set_means(const table& value) {
        set_means_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);
    void set_eigenvalues_impl(const table&);
    void set_eigenvectors_impl(const table&);
    void set_variances_impl(const table&);
    void set_means_impl(const table&);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;

} // namespace oneapi::dal::pca

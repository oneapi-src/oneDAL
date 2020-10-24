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
template <typename Task = task::by_default>
class infer_input_impl;

template <typename Task = task::by_default>
class infer_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class ONEDAL_EXPORT infer_input : public base {
public:
    using task_t = Task;
    infer_input(const model<task_t>& trained_model, const table& data);

    model<task_t> get_model() const;

    auto& set_model(const model<task_t>& value) {
        set_model_impl(value);
        return *this;
    }

    table get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<task_t>& value);
    void set_data_impl(const table& value);

    dal::detail::pimpl<detail::infer_input_impl<task_t>> impl_;
};

template <typename Task = task::by_default>
class ONEDAL_EXPORT infer_result {
public:
    using task_t = Task;
    infer_result();

    table get_transformed_data() const;

    auto& set_transformed_data(const table& value) {
        set_transformed_data_impl(value);
        return *this;
    }

private:
    void set_transformed_data_impl(const table&);

    dal::detail::pimpl<detail::infer_result_impl<task_t>> impl_;
};

} // namespace oneapi::dal::pca

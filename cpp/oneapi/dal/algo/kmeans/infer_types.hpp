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

namespace oneapi::dal::kmeans {

namespace detail {
class infer_input_impl;
class infer_result_impl;
} // namespace detail

class ONEAPI_DAL_EXPORT infer_input : public base {
public:
    infer_input(const model& trained_model, const table& data);

    model get_model() const;

    auto& set_model(const model& value) {
        set_model_impl(value);
        return *this;
    }

    table get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model& value);
    void set_data_impl(const table& value);

    dal::detail::pimpl<detail::infer_input_impl> impl_;
};

class ONEAPI_DAL_EXPORT infer_result {
public:
    infer_result();

    table get_labels() const;
    double get_objective_function_value() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    auto& set_objective_function_value(double value) {
        set_objective_function_value_impl(value);
        return *this;
    }

private:
    void set_labels_impl(const table&);
    void set_objective_function_value_impl(double);

    dal::detail::pimpl<detail::infer_result_impl> impl_;
};

} // namespace oneapi::dal::kmeans

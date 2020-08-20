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

#include "oneapi/dal/algo/svm/common.hpp"

namespace oneapi::dal::svm {

namespace detail {
class train_input_impl;
class train_result_impl;
} // namespace detail

class ONEAPI_DAL_EXPORT train_input : public base {
public:
    train_input(const table& data, const table& labels, const table& weights = table{});

    table get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    table get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    table get_weights() const;

    auto& set_weights(const table& value) {
        set_weights_impl(value);
        return *this;
    }

private:
    void set_data_impl(const table& value);
    void set_labels_impl(const table& value);
    void set_weights_impl(const table& value);

    dal::detail::pimpl<detail::train_input_impl> impl_;
};

class ONEAPI_DAL_EXPORT train_result : public base {
public:
    train_result();

    model get_model() const;
    table get_support_vectors() const;
    table get_support_indices() const;
    table get_coeffs() const;
    double get_bias() const;
    std::int64_t get_support_vector_count() const;

    auto& set_model(const model& value) {
        set_model_impl(value);
        return *this;
    }

    auto& set_support_vectors(const table& value) {
        set_support_vectors_impl(value);
        return *this;
    }

    auto& set_support_indices(const table& value) {
        set_support_indices_impl(value);
        return *this;
    }

    auto& set_coeffs(const table& value) {
        set_coeffs_impl(value);
        return *this;
    }

    auto& set_bias(double value) {
        set_bias_impl(value);
        return *this;
    }

    auto& set_support_vector_count(std::int64_t value) {
        set_support_vector_count_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model&);
    void set_support_vectors_impl(const table&);
    void set_support_indices_impl(const table&);
    void set_coeffs_impl(const table&);
    void set_bias_impl(double);
    void set_support_vector_count_impl(std::int64_t);

    dal::detail::pimpl<detail::train_result_impl> impl_;
};

} // namespace oneapi::dal::svm

/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/linear_regression/common.hpp"

#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::linear_regression {

template <typename Task>
class detail::v1::model_impl : public base {
public:
    model_impl() = default;

    virtual const table& get_betas() const = 0;
};

namespace backend {

template <typename Task>
using model_impl = detail::model_impl<Task>;

using norm_eq_proto = ONEDAL_SERIALIZABLE(linear_regression_norm_eq_model_impl_id);

template <typename Task>
class norm_eq_model_impl : public norm_eq_proto, public model_impl<Task> {
public:
    norm_eq_model_impl() = default;

    norm_eq_model_impl(const table& betas) : betas_(betas) {}

    void serialize(dal::detail::output_archive& ar) const override {
        ar(betas_);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(betas_);
    }

    const table& get_betas() const override {
        return betas_;
    }

private:
    table betas_;
};

} // namespace backend
} // namespace oneapi::dal::linear_regression

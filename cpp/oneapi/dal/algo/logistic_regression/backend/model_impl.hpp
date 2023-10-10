/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/logistic_regression/common.hpp"

#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::logistic_regression {

using dense_batch_proto = ONEDAL_SERIALIZABLE(logistic_regression_model_impl_id);

template <typename Task>
class detail::v1::model_impl : public dense_batch_proto {
public:
    model_impl() = default;

    model_impl(const table& packed_coefficients) : packed_coefficients_(packed_coefficients) {}

    void serialize(dal::detail::output_archive& ar) const {
        ar(packed_coefficients_);
    }

    void deserialize(dal::detail::input_archive& ar) {
        ar(packed_coefficients_);
    }

    const table& get_packed_coefficients() const {
        return packed_coefficients_;
    }

    void set_packed_coefficients(const table& v) {
        this->packed_coefficients_ = v;
    }

private:
    table packed_coefficients_;
};

} // namespace oneapi::dal::logistic_regression

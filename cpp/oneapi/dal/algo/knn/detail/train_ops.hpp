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

#include "oneapi/dal/algo/knn/train_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::knn::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT train_ops_dispatcher {
    train_result operator()(const Context&, const descriptor_base&, const train_input&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = train_input;
    using result_t = train_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const train_input& input) const {
        if (!(input.get_data().has_data())) {
            throw domain_error("Input data should not be empty");
        }
        if (!(input.get_labels().has_data())) {
            throw domain_error("Input labels should not be empty");
        }
        if (input.get_labels().get_column_count() != 1) {
            throw domain_error("Labels should contain a single column");
        }
        if (!(input.get_labels().get_row_count() == input.get_data().get_row_count())) {
            throw domain_error("Number of labels should match number of rows in data");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const train_input& input,
                              const train_result& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const train_input& input) const {
        check_preconditions(desc, input);
        const auto result = train_ops_dispatcher<Context, float_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
}; // namespace oneapi::dal::knn::detail

} // namespace oneapi::dal::knn::detail

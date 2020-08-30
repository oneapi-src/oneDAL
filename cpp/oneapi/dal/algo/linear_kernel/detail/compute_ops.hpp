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

#include "oneapi/dal/algo/linear_kernel/compute_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::linear_kernel::detail {

template <typename Context, typename... Options>
struct compute_ops_dispatcher {
    compute_result operator()(const Context&, const descriptor_base&, const compute_input&) const;
};

template <typename Descriptor>
struct compute_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = compute_input;
    using result_t = compute_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const compute_input& input) const {
        if (!(input.get_x().has_data())) {
            throw domain_error("Input x should not be empty");
        }
        if (!(input.get_y().has_data())) {
            throw domain_error("Input y should not be empty");
        }
        if (input.get_x().get_column_count() != input.get_y().get_column_count()) {
            throw invalid_argument("Input x column_count should be equal to y column_count");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const compute_input& input,
                              const compute_result& result) const {
        if (!(result.get_values().has_data())) {
            throw domain_error("Result values should not be empty");
        }
        if (input.get_x().get_row_count() != result.get_values().get_row_count()) {
            throw internal_error("Input x row_count should be equal to values row_count");
        }
        if (input.get_y().get_row_count() != result.get_values().get_column_count()) {
            throw internal_error("Input y row_count should be equal to values col_count");
        }
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const compute_input& input) const {
        check_preconditions(desc, input);
        const auto result = compute_ops_dispatcher<Context, float_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::linear_kernel::detail

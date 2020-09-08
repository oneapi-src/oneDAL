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

#include "oneapi/dal/algo/svm/train_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT train_ops_dispatcher {
    train_result operator()(const Context&, const descriptor_base&, const train_input&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using kernel_t = typename Descriptor::kernel_t;
    using input_t = train_input;
    using result_t = train_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const train_input& input) const {
        if (!(input.get_data().has_data())) {
            throw invalid_argument("Input data should not be empty");
        }
        if (!(input.get_labels().has_data())) {
            throw invalid_argument("Input labels should not be empty");
        }
        if (input.get_data().get_row_count() != input.get_labels().get_row_count()) {
            throw invalid_argument("Input data row_count should be equal to labels row_count");
        }
        if (input.get_weights().has_data()) {
            if (input.get_data().get_row_count() != input.get_weights().get_row_count()) {
                throw invalid_argument("Input data row_count should be equal to weights row_count");
            }
        }
        if (!(params.get_kernel_impl()->get_impl())) {
            throw invalid_argument("Input kernel should be not be empty");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const train_input& input,
                              const train_result& result) const {
        if (result.get_support_vector_count() < 0 ||
            result.get_support_vector_count() > input.get_data().get_row_count()) {
            throw internal_error(
                "Result support_vector_count should be >= 0 and <= input data row_count");
        }
        if (!(result.get_support_vectors().has_data())) {
            throw internal_error("Result support_vectors should not be empty");
        }
        if (!(result.get_support_indices().has_data())) {
            throw internal_error("Result support_indices should not be empty");
        }
        if (!(result.get_coeffs().has_data())) {
            throw internal_error("Result coeffs should not be empty");
        }
        if (result.get_support_vectors().get_column_count() !=
            input.get_data().get_column_count()) {
            throw internal_error(
                "Result support_vectors column_count should be equal to input data column_count");
        }
        if (result.get_support_vectors().get_row_count() != result.get_support_vector_count()) {
            throw internal_error(
                "Result support_vectors row_count should be equal to result support_vector_count");
        }
        if (result.get_support_indices().get_row_count() != result.get_support_vector_count()) {
            throw internal_error(
                "Result support_indices row_count should be equal to result support_vector_count");
        }
        if (result.get_coeffs().get_row_count() != result.get_support_vector_count()) {
            throw internal_error(
                "Result coeffs row_count should be equal to result support_vector_count");
        }
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const train_input& input) const {
        check_preconditions(desc, input);
        const auto result =
            train_ops_dispatcher<Context, float_t, task_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::svm::detail

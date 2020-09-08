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

#include "oneapi/dal/algo/svm/infer_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT infer_ops_dispatcher {
    infer_result operator()(const Context&, const descriptor_base&, const infer_input&) const;
};

template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = method::by_default;
    using input_t = infer_input;
    using result_t = infer_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const infer_input& input) const {
        if (!(input.get_data().has_data())) {
            throw domain_error("Input data should not be empty");
        }
        if (input.get_model().get_support_vector_count() < 0) {
            throw invalid_argument("Input model support_vector_count should be >= 0");
        }
        if (!(input.get_model().get_support_vectors().has_data())) {
            throw domain_error("Input model support_vectors should not be empty");
        }
        if (!(input.get_model().get_coeffs().has_data())) {
            throw domain_error("Input model coeffs should not be empty");
        }
        if (input.get_model().get_support_vectors().get_column_count() !=
            input.get_data().get_column_count()) {
            throw invalid_argument(
                "Input model support_vectors column_count should be equal to input data column_count");
        }
        if (input.get_model().get_support_vectors().get_row_count() !=
            input.get_model().get_support_vector_count()) {
            throw invalid_argument(
                "Input model support_vectors row_count should be equal to input model support_vector_count");
        }
        if (input.get_model().get_coeffs().get_row_count() !=
            input.get_model().get_support_vector_count()) {
            throw invalid_argument(
                "Input model coeffs row_count should be equal to input model support_vector_count");
        }
        if (!(params.get_kernel_impl()->get_impl())) {
            throw domain_error("Input kernel should be not be empty");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const infer_input& input,
                              const infer_result& result) const {
        if (!(result.get_labels().has_data())) {
            throw domain_error("Relult labels should not be empty");
        }
        if (!(result.get_decision_function().has_data())) {
            throw domain_error("Relult decision_function should not be empty");
        }
        if (result.get_decision_function().get_row_count() != result.get_labels().get_row_count()) {
            throw internal_error(
                "Relult decision_function row_count should be equal labels row_count");
        }
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const infer_input& input) const {
        check_preconditions(desc, input);
        const auto result =
            infer_ops_dispatcher<Context, float_t, task_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::svm::detail

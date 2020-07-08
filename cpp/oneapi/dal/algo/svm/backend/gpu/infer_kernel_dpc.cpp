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

#define DAAL_SYCL_INTERFACE

#include "oneapi/dal/algo/svm/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/interop_model.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include <daal/src/algorithms/svm/oneapi/svm_predict_kernel_oneapi.h>

namespace oneapi::dal::svm::backend {

using std::int64_t;
using dal::backend::context_gpu;

namespace daal_svm             = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop              = dal::backend::interop;

template <typename Float>
using daal_svm_predict_kernel_t =
    daal_svm::prediction::internal::SVMPredictImplOneAPI<daal_svm::prediction::defaultDense, Float>;

template <typename Float>
static infer_result call_daal_kernel(const context_gpu& ctx,
                                     const descriptor_base& desc,
                                     const model& trained_model,
                                     const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count             = data.get_row_count();
    const int64_t column_count          = data.get_column_count();
    const int64_t support_vectors_count = trained_model.get_support_vector_count();

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    auto arr_support_vectors =
        row_accessor<const Float>{ trained_model.get_support_vectors() }.pull(queue);
    auto arr_coefficients =
        row_accessor<const Float>{ trained_model.get_coefficients() }.pull(queue);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_support_vectors =
        interop::convert_to_daal_sycl_homogen_table(queue,
                                                    arr_support_vectors,
                                                    support_vectors_count,
                                                    column_count);
    const auto daal_coefficients =
        interop::convert_to_daal_sycl_homogen_table(queue,
                                                    arr_coefficients,
                                                    support_vectors_count,
                                                    1);

    auto daal_model = daal_model_builder{}
                          .set_support_vectors(daal_support_vectors)
                          .set_coefficients(daal_coefficients)
                          .set_bias(trained_model.get_bias());

    auto kernel_impl       = desc.get_kernel_impl()->get_impl();
    const auto daal_kernel = kernel_impl->get_daal_kernel_function();

    daal_svm::Parameter daal_parameter(daal_kernel);

    auto arr_decision_function = array<Float>::empty(queue, row_count * 1);
    const auto daal_decision_function =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_decision_function, row_count, 1);

    interop::status_to_exception(daal_svm_predict_kernel_t<Float>().compute(daal_data,
                                                                            &daal_model,
                                                                            *daal_decision_function,
                                                                            &daal_parameter));

    auto arr_label = array<Float>::empty(queue, row_count * 1);
    auto event     = queue.submit([&](sycl::handler& cgh) {
        auto labels            = arr_label.get_mutable_data();
        auto decision_function = arr_decision_function.get_data();
        cgh.parallel_for(sycl::range<1>(arr_label.get_count()), [=](sycl::id<1> idx) {
            labels[idx[0]] = decision_function[idx[0]] >= 0 ? Float(1.0) : Float(-1.0);
        });
    });
    event.wait();

    return infer_result()
        .set_decision_function(
            homogen_table_builder{}.reset(arr_decision_function, row_count, 1).build())
        .set_labels(homogen_table_builder{}.reset(arr_label, row_count, 1).build());
}

template <typename Float>
static infer_result infer(const context_gpu& ctx,
                          const descriptor_base& desc,
                          const infer_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_gpu<Float, task::classification, method::by_default> {
    infer_result operator()(const context_gpu& ctx,
                            const descriptor_base& desc,
                            const infer_input& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::classification, method::by_default>;
template struct infer_kernel_gpu<double, task::classification, method::by_default>;

} // namespace oneapi::dal::svm::backend

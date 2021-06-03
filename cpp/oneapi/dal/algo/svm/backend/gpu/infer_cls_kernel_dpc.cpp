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

#include <daal/src/algorithms/svm/oneapi/svm_predict_kernel_oneapi.h>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/transfer.hpp"

namespace oneapi::dal::svm::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = infer_input<task::classification>;
using result_t = infer_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_svm = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_svm_predict_kernel_t =
    daal_svm::prediction::internal::SVMPredictImplOneAPI<daal_svm::prediction::defaultDense, Float>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const model_t& trained_model,
                                 const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::uint64_t class_count = desc.get_class_count();
    if (class_count > 2) {
        throw unimplemented(dal::detail::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const std::int64_t row_count = data.get_row_count();

    const auto daal_data = interop::convert_to_daal_table(queue, data);
    const auto daal_support_vectors =
        interop::convert_to_daal_table(queue, trained_model.get_support_vectors());
    const auto daal_coeffs = interop::convert_to_daal_table(queue, trained_model.get_coeffs());

    const auto biases = trained_model.get_biases();
    const auto biases_acc = row_accessor<const Float>{ biases }.pull();
    const double bias = biases_acc[0];

    auto daal_model = daal_model_builder{}
                          .set_support_vectors(daal_support_vectors)
                          .set_coeffs(daal_coeffs)
                          .set_bias(bias);

    auto kernel_impl = detail::get_kernel_function_impl(desc);
    if (!kernel_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }
    const bool is_dense{ data.get_kind() == homogen_table::kind() };
    const auto daal_kernel = kernel_impl->get_daal_kernel_function(is_dense);

    daal_svm::Parameter daal_parameter(daal_kernel);

    auto arr_decision_func = array<Float>::empty(queue, row_count * 1, sycl::usm::alloc::device);
    const auto daal_decision_function =
        interop::convert_to_daal_table(queue, arr_decision_func, row_count, 1);

    interop::status_to_exception(daal_svm_predict_kernel_t<Float>().compute(daal_data,
                                                                            &daal_model,
                                                                            *daal_decision_function,
                                                                            &daal_parameter));

    const auto arr_decision_func_host = dal::backend::to_host_sync(arr_decision_func);

    // TODO: rework with help dpcpp code
    auto arr_label = array<Float>::empty(row_count * 1);
    auto label_data = arr_label.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; ++i) {
        label_data[i] = arr_decision_func_host[i] >= 0 ? trained_model.get_second_class_label()
                                                       : trained_model.get_first_class_label();
    }

    return result_t()
        .set_decision_function(dal::detail::homogen_table_builder{}
                                   .reset(arr_decision_func_host, row_count, 1)
                                   .build())
        .set_labels(dal::detail::homogen_table_builder{}.reset(arr_label, row_count, 1).build());
}

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::classification> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::nu_classification> {
    infer_result<task::nu_classification> operator()(
        const context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& desc,
        const infer_input<task::nu_classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_nu_classification_task_is_not_implemented_for_gpu());
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::classification>;
template struct infer_kernel_gpu<double, method::by_default, task::classification>;
template struct infer_kernel_gpu<float, method::by_default, task::nu_classification>;
template struct infer_kernel_gpu<double, method::by_default, task::nu_classification>;

} // namespace oneapi::dal::svm::backend

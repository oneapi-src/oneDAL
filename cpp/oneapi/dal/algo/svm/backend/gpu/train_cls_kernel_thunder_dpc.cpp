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

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/model_interop.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/utils.hpp"

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/svm/oneapi/svm_train_thunder_kernel_oneapi.h>

namespace oneapi::dal::svm::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = train_input<task::classification>;
using result_t = train_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_svm = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_svm_thunder_kernel_t =
    daal_svm::training::internal::SVMTrainOneAPI<Float, daal_svm::training::thunder>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& labels) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::uint64_t class_count = desc.get_class_count();
    if (class_count > 2) {
        throw unimplemented(dal::detail::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const std::int64_t column_count = data.get_column_count();

    const binary_label_t<Float> old_unique_labels = get_unique_labels<Float>(queue, labels);
    const auto new_labels =
        convert_binary_labels(queue, labels, { Float(-1.0), Float(1.0) }, old_unique_labels);
    const auto daal_labels = interop::convert_to_daal_table(queue, new_labels);

    auto kernel_impl = detail::get_kernel_function_impl(desc);
    if (!kernel_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }

    const bool is_dense{ data.get_kind() == homogen_table::kind() };
    const auto daal_kernel = kernel_impl->get_daal_kernel_function(is_dense);

    const std::uint64_t cache_megabyte = static_cast<std::uint64_t>(desc.get_cache_size());
    constexpr std::uint64_t megabyte = 1024 * 1024;
    dal::detail::check_mul_overflow(cache_megabyte, megabyte);
    const std::uint64_t cache_byte = cache_megabyte * megabyte;

    daal_svm::training::internal::KernelParameter daal_svm_parameter;
    daal_svm_parameter.kernel = daal_kernel;
    daal_svm_parameter.C = desc.get_c();
    daal_svm_parameter.accuracyThreshold = desc.get_accuracy_threshold();
    daal_svm_parameter.tau = desc.get_tau();
    daal_svm_parameter.maxIterations =
        dal::detail::integral_cast<std::size_t>(desc.get_max_iteration_count());
    daal_svm_parameter.doShrinking = desc.get_shrinking();
    daal_svm_parameter.cacheSize = cache_byte;

    const auto daal_data = interop::convert_to_daal_table(queue, data);
    auto daal_model = daal_svm::Model::create<Float>(column_count);
    interop::status_to_exception(daal_svm_thunder_kernel_t<Float>().compute(daal_data,
                                                                            *daal_labels,
                                                                            daal_model.get(),
                                                                            daal_svm_parameter));
    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getSupportIndices());

    auto trained_model = convert_from_daal_model<task::classification, Float>(*daal_model)
                             .set_first_class_label(old_unique_labels.first)
                             .set_second_class_label(old_unique_labels.second);

    return result_t().set_model(trained_model).set_support_indices(table_support_indices);
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float>
struct train_kernel_gpu<Float, method::thunder, task::classification> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template <typename Float>
struct train_kernel_gpu<Float, method::thunder, task::nu_classification> {
    train_result<task::nu_classification> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& params,
        const train_input<task::nu_classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::nu_svm_thunder_method_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::thunder, task::classification>;
template struct train_kernel_gpu<double, method::thunder, task::classification>;
template struct train_kernel_gpu<float, method::thunder, task::nu_classification>;
template struct train_kernel_gpu<double, method::thunder, task::nu_classification>;

} // namespace oneapi::dal::svm::backend

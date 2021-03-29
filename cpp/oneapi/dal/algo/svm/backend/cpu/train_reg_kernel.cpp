/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <daal/src/algorithms/svm/svm_train_boser_kernel.h>
#include <daal/src/algorithms/svm/svm_train_thunder_kernel.h>

#include "algorithms/svm/svm_train.h"

#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/model_interop.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/utils.hpp"
#include "oneapi/dal/algo/svm/backend/model_impl.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::svm::backend {

using dal::backend::context_cpu;
using model_t = model<task::regression>;
using input_t = train_input<task::regression>;
using result_t = train_result<task::regression>;
using descriptor_t = detail::descriptor_base<task::regression>;

namespace daal_svm = daal::algorithms::svm;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu, typename Method>
using daal_svm_kernel_t =
    daal_svm::training::internal::SVMTrainImpl<to_daal_method<Method>::value, Float, Cpu>;

template <typename Float, typename Method>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& labels,
                                 const table& weights) {
    const std::int64_t column_count = data.get_column_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_labels = interop::convert_to_daal_table<Float>(labels);
    const auto daal_weights = interop::convert_to_daal_table<Float>(weights);

    auto kernel_impl = detail::get_kernel_function_impl(desc);
    if (!kernel_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }
    const auto daal_kernel = kernel_impl->get_daal_kernel_function();

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
    daal_svm_parameter.epsilon = desc.get_epsilon();
    daal_svm_parameter.svmType = daal_svm::training::internal::SvmType::regression;

    auto daal_model = daal_svm::Model::create<Float>(column_count);
    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return daal_svm_kernel_t<
                   Float,
                   oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                   Method>()
            .compute(daal_data, daal_weights, *daal_labels, daal_model.get(), daal_svm_parameter);
    }));
    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getSupportIndices());

    auto trained_model = convert_from_daal_model<task::regression, Float>(*daal_model);
    return result_t().set_model(trained_model).set_support_indices(table_support_indices);
}

template <typename Float, typename Method>
static result_t train(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float, Method>(ctx,
                                           desc,
                                           input.get_data(),
                                           input.get_labels(),
                                           input.get_weights());
}

template <typename Float, typename Method>
struct train_kernel_cpu<Float, Method, task::regression> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float, Method>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::thunder, task::regression>;
template struct train_kernel_cpu<double, method::thunder, task::regression>;

} // namespace oneapi::dal::svm::backend

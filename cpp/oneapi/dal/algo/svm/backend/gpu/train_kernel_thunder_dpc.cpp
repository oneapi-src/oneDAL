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
#define DAAL_SYCL_INTERFACE_USM
#define DAAL_SYCL_INTERFACE_REVERSED_RANGE

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/interop_model.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/utils.hpp"

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/svm/oneapi/svm_train_thunder_kernel_oneapi.h>

namespace oneapi::dal::svm::backend {

using std::int64_t;
using dal::backend::context_gpu;

namespace daal_svm = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_svm_thunder_kernel_t = daal_svm::training::internal::
    SVMTrainOneAPI<Float, daal_svm::Parameter, daal_svm::training::thunder>;

template <typename Float>
static train_result call_daal_kernel(const context_gpu& ctx,
                                     const descriptor_base& desc,
                                     const table& data,
                                     const table& labels) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    auto arr_label = row_accessor<const Float>{ labels }.pull(queue);

    binary_label_t<Float> unique_label;
    auto arr_new_label =
        convert_labels(queue, arr_label, { Float(-1.0), Float(1.0) }, unique_label);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_labels =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_new_label, row_count, 1);

    auto kernel_impl = desc.get_kernel_impl()->get_impl();
    const auto daal_kernel = kernel_impl->get_daal_kernel_function();
    daal_svm::Parameter daal_parameter(
        daal_kernel,
        desc.get_c(),
        desc.get_accuracy_threshold(),
        desc.get_tau(),
        desc.get_max_iteration_count(),
        int64_t(desc.get_cache_size() * 1024 * 1024), // DAAL get in bytes
        desc.get_shrinking());

    auto daal_model = daal_svm::Model::create<Float>(column_count);
    interop::status_to_exception(daal_svm_thunder_kernel_t<Float>().compute(daal_data,
                                                                            *daal_labels,
                                                                            daal_model.get(),
                                                                            &daal_parameter));
    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getSupportIndices());

    auto trained_model = convert_from_daal_model<Float>(*daal_model)
                             .set_first_class_label(unique_label.first)
                             .set_second_class_label(unique_label.second);

    return train_result().set_model(trained_model).set_support_indices(table_support_indices);
}

template <typename Float>
static train_result train(const context_gpu& ctx,
                          const descriptor_base& desc,
                          const train_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float>
struct train_kernel_gpu<Float, task::classification, method::thunder> {
    train_result operator()(const dal::backend::context_gpu& ctx,
                            const descriptor_base& desc,
                            const train_input& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, task::classification, method::thunder>;
template struct train_kernel_gpu<double, task::classification, method::thunder>;

} // namespace oneapi::dal::svm::backend

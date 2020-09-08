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

#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h>

#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_interop.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_knn_brute_force_kernel_t =
    daal_knn::prediction::internal::KNNClassificationPredictKernelUCAPI<Float>;

template <typename Float>
static infer_result call_daal_kernel(const context_gpu& ctx,
                                     const descriptor_base& desc,
                                     const table& data,
                                     const model& m) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    auto arr_labels = array<Float>::empty(queue, 1 * row_count);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_labels =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_labels, row_count, 1);

    daal_knn::Parameter daal_parameter(
        desc.get_class_count(),
        desc.get_neighbor_count(),
        desc.get_data_use_in_model() ? daal_knn::doUse : daal_knn::doNotUse);

    interop::status_to_exception(daal_knn_brute_force_kernel_t<Float>().compute(
        daal_data.get(),
        dal::detail::get_impl<detail::model_impl>(m).get_interop()->get_daal_model().get(),
        daal_labels.get(),
        &daal_parameter));

    return infer_result().set_labels(
        dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build());
}

template <typename Float>
static infer_result infer(const context_gpu& ctx,
                          const descriptor_base& desc,
                          const infer_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float>
struct infer_kernel_gpu<Float, method::brute_force> {
    infer_result operator()(const context_gpu& ctx,
                            const descriptor_base& desc,
                            const infer_input& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force>;
template struct infer_kernel_gpu<double, method::brute_force>;

} // namespace oneapi::dal::knn::backend

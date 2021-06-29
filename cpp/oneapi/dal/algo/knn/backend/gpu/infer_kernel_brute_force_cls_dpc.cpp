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

#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h>

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_knn_brute_force_kernel_t =
    daal_knn::prediction::internal::KNNClassificationPredictKernelUCAPI<Float>;

template <typename Float>
static infer_result<task::classification> call_daal_kernel(const context_gpu& ctx,
                                                           const descriptor_t& desc,
                                                           const table& data,
                                                           const model<task::classification> m) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t row_count = data.get_row_count();
    auto arr_labels = array<Float>::empty(queue, 1 * row_count, sycl::usm::alloc::device);

    const auto daal_data = interop::convert_to_daal_table(queue, data);
    const auto daal_labels = interop::convert_to_daal_table(queue, arr_labels, row_count, 1);

    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }
    else if (distance_impl->get_daal_distance_type() != detail::v1::daal_distance_t::minkowski ||
             distance_impl->get_degree() != 2.0) {
        throw internal_error{ dal::detail::error_messages::distance_is_not_supported_for_gpu() };
    }

    interop::status_to_exception(daal_knn_brute_force_kernel_t<Float>().compute(
        daal_data.get(),
        dal::detail::get_impl(m).get_interop()->get_daal_model().get(),
        daal_labels.get(),
        &daal_parameter));

    return infer_result<task::classification>().set_labels(
        dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build());
}

template <typename Float>
static infer_result<task::classification> infer(const context_gpu& ctx,
                                                const descriptor_t& desc,
                                                const infer_input<task::classification>& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float>
struct infer_kernel_gpu<Float, method::brute_force, task::classification> {
    infer_result<task::classification> operator()(
        const context_gpu& ctx,
        const descriptor_t& desc,
        const infer_input<task::classification>& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force, task::classification>;
template struct infer_kernel_gpu<double, method::brute_force, task::classification>;

} // namespace oneapi::dal::knn::backend

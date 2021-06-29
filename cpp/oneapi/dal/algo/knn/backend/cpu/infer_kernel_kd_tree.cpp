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

#include <daal/src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch.h>

#include "oneapi/dal/algo/knn/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_knn = daal::algorithms::kdtree_knn_classification;
namespace daal_classifier = daal::algorithms::classifier;

namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_kd_tree_kernel_t = daal_knn::prediction::internal::
    KNNClassificationPredictKernel<Float, daal_knn::prediction::defaultDense, Cpu>;

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_cpu &ctx,
                                           const descriptor_t &desc,
                                           const table &data,
                                           model<Task> m) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();

    auto arr_labels = array<Float>{};
    auto arr_indices = array<std::int64_t>{};
    auto arr_distance = array<Float>{};

    auto daal_labels = daal::data_management::NumericTablePtr();
    auto daal_indices = daal::data_management::NumericTablePtr();
    auto daal_distance = daal::data_management::NumericTablePtr();

    const std::int64_t dummy_seed = 777;
    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        dal::detail::integral_cast<int>(dummy_seed),
        data_use_in_model);

    const auto daal_voting_mode = convert_to_daal_kdtree_voting_mode(desc.get_voting_mode());
    daal_parameter.voteWeights = daal_voting_mode;

    if constexpr (std::is_same_v<Task, task::search>) {
        daal_parameter.resultsToEvaluate = daal_classifier::none;
        daal_parameter.resultsToCompute =
            daal_knn::computeDistances | daal_knn::computeIndicesOfNeighbors;

        arr_indices.reset(neighbor_count * row_count);
        daal_indices =
            interop::convert_to_daal_homogen_table(arr_indices, row_count, neighbor_count);

        arr_distance.reset(neighbor_count * row_count);
        daal_distance =
            interop::convert_to_daal_homogen_table(arr_distance, row_count, neighbor_count);
    }
    else {
        arr_labels.reset(1 * row_count);
        daal_labels = interop::convert_to_daal_homogen_table(arr_labels, row_count, 1);
    }

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    interop::status_to_exception(interop::call_daal_kernel<Float, daal_knn_kd_tree_kernel_t>(
        ctx,
        daal_data.get(),
        dal::detail::get_impl(m).get_interop()->get_daal_model().get(),
        daal_labels.get(),
        daal_indices.get(),
        daal_distance.get(),
        &daal_parameter));

    auto result = infer_result<Task>{};
    if constexpr (std::is_same_v<Task, task::search>) {
        result = result
                     .set_indices(dal::detail::homogen_table_builder{}
                                      .reset(arr_indices, row_count, neighbor_count)
                                      .build())
                     .set_distances(dal::detail::homogen_table_builder{}
                                        .reset(arr_distance, row_count, neighbor_count)
                                        .build());
    }
    else {
        result = result.set_labels(
            dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build());
    }

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_cpu &ctx,
                                const descriptor_t &desc,
                                const infer_input<Task> &input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, method::kd_tree, Task> {
    infer_result<Task> operator()(const context_cpu &ctx,
                                  const descriptor_t &desc,
                                  const infer_input<Task> &input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::kd_tree, task::classification>;
template struct infer_kernel_cpu<double, method::kd_tree, task::classification>;
template struct infer_kernel_cpu<float, method::kd_tree, task::search>;
template struct infer_kernel_cpu<double, method::kd_tree, task::search>;

} // namespace oneapi::dal::knn::backend

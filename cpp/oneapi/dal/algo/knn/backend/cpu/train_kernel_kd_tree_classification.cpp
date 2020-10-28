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

#include <daal/src/algorithms/k_nearest_neighbors/kdtree_knn_classification_train_kernel.h>
#include <src/algorithms/k_nearest_neighbors/kdtree_knn_classification_model_impl.h>

#include "oneapi/dal/algo/knn/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace daal_knn = daal::algorithms::kdtree_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_kd_tree_kernel_t = daal_knn::training::internal::
    KNNClassificationTrainBatchKernel<Float, daal_knn::training::defaultDense, Cpu>;

template <typename Float>
static train_result<task::classification> call_daal_kernel(
    const context_cpu& ctx,
    const descriptor_base<task::classification>& desc,
    const table& data,
    const table& labels) {
    using daal_model_interop_t = model_interop;
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull();
    auto arr_labels = row_accessor<const Float>{ labels }.pull();

    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_labels = interop::convert_to_daal_homogen_table(arr_labels, row_count, 1);

    const std::int64_t dummy_seed = 777;
    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<size_t>(desc.get_class_count()),
        dal::detail::integral_cast<size_t>(desc.get_neighbor_count()),
        dal::detail::integral_cast<int>(dummy_seed),
        data_use_in_model);

    Status status;
    const daal::algorithms::classifier::ModelPtr model_ptr =
        daal_knn::Model::create(column_count, &status);
    interop::status_to_exception(status);

    auto knn_model = static_cast<daal_knn::Model*>(model_ptr.get());
    const bool copy_data_labels = true;
    knn_model->impl()->setData<Float>(daal_data, copy_data_labels);
    knn_model->impl()->setLabels<Float>(daal_labels, copy_data_labels);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_knn_kd_tree_kernel_t>(ctx,
                                                                    daal_data.get(),
                                                                    daal_labels.get(),
                                                                    knn_model,
                                                                    *daal_parameter.engine.get()));

    auto interop = new daal_model_interop_t(model_ptr);
    const auto model_impl = std::make_shared<detail::model_impl>(interop);
    return train_result<task::classification>().set_model(
        dal::detail::pimpl_accessor::make<model<task::classification>>(model_impl));
}

template <typename Float>
static train_result<task::classification> train(const context_cpu& ctx,
                                                const descriptor_base<task::classification>& desc,
                                                const train_input<task::classification>& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float>
struct train_kernel_cpu<Float, method::kd_tree, task::classification> {
    train_result<task::classification> operator()(
        const context_cpu& ctx,
        const descriptor_base<task::classification>& desc,
        const train_input<task::classification>& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::kd_tree, task::classification>;
template struct train_kernel_cpu<double, method::kd_tree, task::classification>;

} // namespace oneapi::dal::knn::backend

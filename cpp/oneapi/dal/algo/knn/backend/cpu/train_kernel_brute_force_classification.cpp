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

#include <daal/src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h>
#include <algorithms/k_nearest_neighbors/bf_knn_classification_model.h>

#include "oneapi/dal/algo/knn/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using daal::services::Status;
using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_bf_kernel_t = daal_knn::training::internal::
    KNNClassificationTrainKernel<Float, Cpu>;

template <typename Float>
static train_result<task::classification> call_daal_kernel(const context_cpu& ctx,
                                                           const descriptor_t& desc,
                                                           const table& data,
                                                           const table& labels) {
    using daal_model_interop_t = model_interop;
    const std::int64_t column_count = data.get_column_count();

    const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);
    const auto daal_labels = interop::copy_to_daal_homogen_table<Float>(labels);

    const auto data_use_in_model = daal_knn::doUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    Status status;
    const auto /* daal_knn::ModelPtr  */model_ptr = daal_knn::ModelPtr(new daal_knn::Model(column_count));
    //TODO: add checker for pointer
    interop::status_to_exception(status);

    // auto knn_model = static_cast<daal_knn::Model*>(model_ptr.get());

    // Data or labels should not be copied, copy is already happened when
    // the tables are converted to NumericTables
    const bool copy_data_labels = data_use_in_model == daal_knn::doNotUse;
    model_ptr->impl()->setData<Float>(daal_data, copy_data_labels);
    model_ptr->impl()->setLabels<Float>(daal_labels, copy_data_labels);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_knn_bf_kernel_t>(ctx,
                                                                daal_data.get(),
                                                                daal_labels.get(),
                                                                model_ptr.get(),
                                                                daal_parameter,
                                                                *daal_parameter.engine));

    auto interop = new daal_model_interop_t(model_ptr);
    const auto model_impl = std::make_shared<model_impl_cls>(interop);
    return train_result<task::classification>().set_model(
        dal::detail::make_private<model<task::classification>>(model_impl));
}

template <typename Float>
static train_result<task::classification> train(const context_cpu& ctx,
                                                const descriptor_t& desc,
                                                const train_input<task::classification>& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float>
struct train_kernel_cpu<Float, method::brute_force, task::classification> {
    train_result<task::classification> operator()(const context_cpu& ctx,
                                  const descriptor_t& desc,
                                  const train_input<task::classification>& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::brute_force, task::classification>;
template struct train_kernel_cpu<double, method::brute_force, task::classification>;

} // namespace oneapi::dal::knn::backend

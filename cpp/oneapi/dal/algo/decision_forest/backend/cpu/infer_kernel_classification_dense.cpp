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

#include <daal/src/algorithms/dtrees/forest/classification/df_classification_predict_dense_default_batch.h>

#include "oneapi/dal/algo/decision_forest/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/decision_forest/backend/interop_helpers.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_cpu;

namespace df  = daal::algorithms::decision_forest;
namespace cls = daal::algorithms::decision_forest::classification;

namespace interop    = dal::backend::interop;
namespace df_interop = dal::backend::interop::decision_forest;

template <typename Float, daal::CpuType Cpu>
using cls_dense_predict_kernel_t =
    cls::prediction::internal::PredictKernel<Float, cls::prediction::defaultDense, Cpu>;

using cls_model_p = cls::ModelPtr;

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const descriptor_base<Task>& desc,
                                           const infer_input<Task>& input) {
    const model<Task>& trained_model = input.get_model();
    const table& data                = input.get_data();
    const int64_t row_count          = data.get_row_count();
    const int64_t column_count       = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull();

    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);

    /* init param for daal kernel */
    auto daal_input = daal::algorithms::classifier::prediction::Input();
    daal_input.set(daal::algorithms::classifier::prediction::data, daal_data);

    auto model_pimpl = dal::detail::pimpl_accessor().get_pimpl(trained_model);
    if (!model_pimpl->is_interop()) {
        // throw exception
        return infer_result<Task>();
    }

    auto pinterop_model =
        static_cast<df_interop::interop_model_impl<Task, cls_model_p>*>(model_pimpl.get());

    daal_input.set(daal::algorithms::classifier::prediction::model, pinterop_model->get_model());

    auto daal_voting_method = df_interop::convert_to_daal_voting_method(desc.get_voting_method());

    auto daal_parameter = cls::prediction::Parameter(desc.get_class_count(), daal_voting_method);

    daal::data_management::NumericTablePtr daal_labels_res;
    daal::data_management::NumericTablePtr daal_labels_prob_res;

    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(infer_result_to_compute::compute_class_labels)) {
        daal_labels_res = interop::allocate_daal_homogen_table<Float>(row_count, 1);
    }

    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(infer_result_to_compute::compute_class_probabilities)) {
        daal_labels_prob_res =
            interop::allocate_daal_homogen_table<Float>(row_count, desc.get_class_count());
    }

    const cls::Model* const daal_model =
        static_cast<cls::Model*>(pinterop_model->get_model().get());
    interop::status_to_exception(interop::call_daal_kernel<Float, cls_dense_predict_kernel_t>(
        ctx,
        daal::services::internal::hostApp(daal_input),
        daal_data.get(),
        daal_model,
        daal_labels_res.get(),
        daal_labels_prob_res.get(),
        desc.get_class_count(),
        daal_voting_method));

    infer_result<Task> res;

    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(infer_result_to_compute::compute_class_labels)) {
        auto table_class_labels = interop::convert_from_daal_homogen_table<Float>(daal_labels_res);
        res.set_labels(table_class_labels);
    }

    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(infer_result_to_compute::compute_class_probabilities)) {
        auto table_class_probs =
            interop::convert_from_daal_homogen_table<Float>(daal_labels_prob_res);
        res.set_probabilities(table_class_probs);
    }

    return res;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_cpu& ctx,
                                const descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input);
}

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, Task, method::dense> {
    infer_result<Task> operator()(const context_cpu& ctx,
                                  const descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, task::classification, method::dense>;
template struct infer_kernel_cpu<double, task::classification, method::dense>;

} // namespace oneapi::dal::decision_forest::backend

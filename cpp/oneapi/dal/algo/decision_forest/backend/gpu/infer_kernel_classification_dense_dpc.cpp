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

#include <daal/src/algorithms/dtrees/forest/classification/oneapi/df_classification_predict_dense_kernel_oneapi.h>

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/decision_forest/backend/interop_helpers.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;

namespace df = daal::algorithms::decision_forest;
namespace cls = daal::algorithms::decision_forest::classification;

namespace interop = dal::backend::interop;
namespace df_interop = dal::backend::interop::decision_forest;

template <typename Float>
using cls_dense_predict_kernel_t =
    cls::prediction::internal::PredictKernelOneAPI<Float, cls::prediction::defaultDense>;

using cls_model_p = cls::ModelPtr;

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const descriptor_base<Task>& desc,
                                           const model<Task>& trained_model,
                                           const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);

    /* init param for daal kernel */
    auto daal_input = daal::algorithms::classifier::prediction::Input();
    daal_input.set(daal::algorithms::classifier::prediction::data, daal_data);

    auto model_pimpl = dal::detail::pimpl_accessor().get_pimpl(trained_model);
    if (!model_pimpl->is_interop()) {
        throw dal::internal_error(
            dal::detail::error_messages::input_model_is_inconsistent_with_kernel_type());
    }

    auto pinterop_model =
        static_cast<df_interop::interop_model_impl<Task, cls_model_p>*>(model_pimpl.get());

    daal_input.set(daal::algorithms::classifier::prediction::model, pinterop_model->get_model());

    auto daal_voting_mode = df_interop::convert_to_daal_voting_mode(desc.get_voting_mode());

    auto daal_parameter = cls::prediction::Parameter(desc.get_class_count(), daal_voting_mode);

    daal::data_management::NumericTablePtr daal_labels;
    daal::data_management::NumericTablePtr daal_labels_prob;
    array<Float> arr_labels;
    array<Float> arr_labels_prob;

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_labels)) {
        arr_labels = array<Float>::empty(queue, 1 * row_count);
        daal_labels = interop::convert_to_daal_sycl_homogen_table(queue, arr_labels, row_count, 1);
    }

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
        arr_labels_prob = array<Float>::empty(queue, desc.get_class_count() * row_count);
        daal_labels_prob = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                       arr_labels_prob,
                                                                       row_count,
                                                                       desc.get_class_count());
    }

    const cls::Model* const daal_model =
        static_cast<cls::Model*>(pinterop_model->get_model().get());
    interop::status_to_exception(
        cls_dense_predict_kernel_t<Float>().compute(daal::services::internal::hostApp(daal_input),
                                                    daal_data.get(),
                                                    daal_model,
                                                    daal_labels.get(),
                                                    daal_labels_prob.get(),
                                                    desc.get_class_count(),
                                                    daal_voting_mode));

    infer_result<Task> res;

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_labels)) {
        res.set_labels(
            dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build());
    }

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
        res.set_probabilities(dal::detail::homogen_table_builder{}
                                  .reset(arr_labels_prob, row_count, desc.get_class_count())
                                  .build());
    }

    return res;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, Task, method::dense> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::classification, method::dense>;
template struct infer_kernel_gpu<double, task::classification, method::dense>;

} // namespace oneapi::dal::decision_forest::backend

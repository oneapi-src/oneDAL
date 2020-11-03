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

#include <daal/src/services/service_algo_utils.h>
#include <daal/src/algorithms/dtrees/forest/classification/oneapi/df_classification_predict_dense_kernel_oneapi.h>

#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = infer_input<task::classification>;
using result_t = infer_result<task::classification>;
using descriptor_t = descriptor_base<task::classification>;

namespace daal_df = daal::algorithms::decision_forest;
namespace daal_df_cls_pred = daal_df::classification::prediction;
namespace interop = dal::backend::interop;

template <typename Float>
using cls_dense_predict_kernel_t =
    daal_df_cls_pred::internal::PredictKernelOneAPI<Float, daal_df_cls_pred::defaultDense>;

static daal_df::classification::ModelPtr get_daal_model(const model_t& trained_model) {
    const model_interop* interop_model = dal::detail::get_impl(trained_model).get_interop();
    if (!interop_model) {
        throw dal::internal_error(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    return static_cast<const model_interop_cls*>(interop_model)->get_model();
}

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const model_t& trained_model,
                                 const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);

    auto daal_model = get_daal_model(trained_model);

    auto daal_input = daal::algorithms::classifier::prediction::Input();
    daal_input.set(daal::algorithms::classifier::prediction::data, daal_data);
    daal_input.set(daal::algorithms::classifier::prediction::model, daal_model);

    auto daal_voting_mode = convert_to_daal_voting_mode(desc.get_voting_mode());

    auto daal_parameter =
        daal_df_cls_pred::Parameter(dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
                                    daal_voting_mode);

    daal::data_management::NumericTablePtr daal_labels;
    daal::data_management::NumericTablePtr daal_labels_prob;
    array<Float> arr_labels;
    array<Float> arr_labels_prob;

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_labels)) {
        arr_labels = array<Float>::empty(queue, 1 * row_count);
        daal_labels = interop::convert_to_daal_sycl_homogen_table(queue, arr_labels, row_count, 1);
    }

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
        dal::detail::check_mul_overflow(desc.get_class_count(), row_count);
        arr_labels_prob = array<Float>::empty(queue, desc.get_class_count() * row_count);
        daal_labels_prob = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                       arr_labels_prob,
                                                                       row_count,
                                                                       desc.get_class_count());
    }

    const daal_df::classification::Model* const daal_model_ptr = daal_model.get();
    interop::status_to_exception(
        cls_dense_predict_kernel_t<Float>().compute(daal::services::internal::hostApp(daal_input),
                                                    daal_data.get(),
                                                    daal_model_ptr,
                                                    daal_labels.get(),
                                                    daal_labels_prob.get(),
                                                    desc.get_class_count(),
                                                    daal_voting_mode));

    result_t res;

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

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, Task, method::dense> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::classification, method::dense>;
template struct infer_kernel_gpu<double, task::classification, method::dense>;

} // namespace oneapi::dal::decision_forest::backend

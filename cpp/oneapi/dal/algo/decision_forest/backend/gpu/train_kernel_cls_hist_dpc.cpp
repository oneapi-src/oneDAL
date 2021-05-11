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

#include <daal/src/services/service_algo_utils.h>
#include <daal/src/algorithms/dtrees/forest/classification/df_classification_model_impl.h>
#include <daal/include/algorithms/decision_forest/decision_forest_classification_training_batch.h>
#include <daal/include/algorithms/decision_forest/decision_forest_classification_training_types.h>
#include <daal/src/algorithms/dtrees/forest/classification/oneapi/df_classification_train_hist_kernel_oneapi.h>

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/df_cls_train_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = train_input<task::classification>;
using result_t = train_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_df = daal::algorithms::decision_forest;
namespace daal_df_cls_train = daal_df::classification::training;
namespace interop = dal::backend::interop;

template <typename Float>
using cls_hist_kernel_t =
    daal_df_cls_train::internal::ClassificationTrainBatchKernelOneAPI<Float,
                                                                      daal_df_cls_train::hist>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& labels) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    df_cls_train_hist_impl<Float> train_hist_impl(queue);
    return train_hist_impl(desc, data, labels);
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::hist, Task> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::hist, task::classification>;
template struct train_kernel_gpu<double, method::hist, task::classification>;

} // namespace oneapi::dal::decision_forest::backend

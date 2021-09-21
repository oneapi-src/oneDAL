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

#include <daal/src/algorithms/linear_regression/linear_regression_train_kernel.h>

#include "oneapi/dal/algo/linear_regression/backend/model_conversion.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

using be = dal::backend;

namespace daal_lr = daal::algorithms::linear_regression;
namespace interop = dal::backend::interop;

constexpr auto daal_method = daal_lr::training::normEqDense;

template <typename Float, daal::CpuType Cpu>
using linear_regression_kernel_t =
    daal_lr::training::internal::BatchKernel<Float, daal_method, Cpu>;


template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& responses) {
    using model_t = model<Task>;

    const bool intercept = desc.get_compute_intercept();

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();
    const auto response_count = responses.get_column_count();

    const auto ext_feature_count = feature_count + intercept;

    auto temp_xty = be::ndarray<Float>::zeros({ response_count, ext_feature_count});
    auto temp_xtx = be::ndarray<Float>::zeros({ ext_feature_count, ext_feature_count });


    return train_result<Task>();
}

template <typename Float, typename Task>
static train_result<Task> train(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::norm_eq, task::regression>;
template struct train_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
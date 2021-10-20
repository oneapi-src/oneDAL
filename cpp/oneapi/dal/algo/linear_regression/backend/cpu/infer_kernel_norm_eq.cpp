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

//#include <daal/src/algorithms/linear_model/linear_model_predict_kernel.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/infer_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/infer_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace be = dal::backend;
//namespace daal_lm = daal::algorithms::linear_model;
namespace interop = dal::backend::interop;

/*constexpr auto daal_method = daal_lm::prediction::Method::defaultDense;

template <typename Float, daal::CpuType Cpu>
using daal_lm_kernel_t =
    daal_lm::prediction::internal::PredictKernel<Float, daal_method, Cpu>;*/


template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data) {
/*    using dal::detail::check_mul_overflow;

    const bool intercept = desc.get_compute_intercept();

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();
    const auto response_count = responses.get_column_count();

    const auto ext_feature_count = feature_count + intercept;

    const auto xtx_size = check_mul_overflow(feature_count, feature_count);
    auto xtx_arr = array<Float>{ xtx_size };

    const auto xty_size = check_mul_overflow(feature_count, response_count);
    auto xty_arr = array<Float>{ xty_size };

    const auto betas_size = check_mul_overflow(ext_feature_count, response_count);
    auto betas_arr = array<Float>{ betas_size};

    auto xtx_daal_table = interop::convert_to_daal_homogen_table(xtx_arr, feature_count, feature_count);
    auto xty_daal_table = interop::convert_to_daal_homogen_table(xty_arr, feature_count, response_count);
    auto betas_daal_table = interop::convert_to_daal_homogen_table(betas_arr, ext_feature_count, betas_count);

    auto x_daal_table = interop::convert_to_daal_table<Float>(data);
    auto y_daal_table = interop::convert_to_daal_table<Float>(resp);

    const auto status = interop::call_daal_kernel<Float, daal_lr_kernel_t>(
        ctx,
        x_daal_table,
        y_daal_table,
        xtx_daal_table,
        xty_daal_table,
        betas_daal_table,
        intercept);

    interop::status_to_exception(status);

    auto betas = homogen_table::wrap(betas_arr, ext_feature_count, response_count);
*/
    auto result = infer_result<Task>();

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data());
}

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, method::norm_eq, Task> {
    infer_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::norm_eq, task::regression>;
template struct infer_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
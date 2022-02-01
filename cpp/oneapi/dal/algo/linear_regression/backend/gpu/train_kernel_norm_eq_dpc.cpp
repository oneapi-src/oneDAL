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

#include <daal/src/algorithms/linear_regression/oneapi/linear_regression_train_kernel_oneapi.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/train_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_gpu;

namespace be = dal::backend;
namespace interop = dal::backend::interop;
namespace daal_lr = daal::algorithms::linear_regression;

constexpr auto daal_method = daal_lr::training::normEqDense;

template <typename Float>
using daal_lr_kernel_t = daal_lr::training::internal::OnlineKernelOneAPI<Float, daal_method>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& resp) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = backend::norm_eq_model_impl<Task>;

    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    bool intp = desc.get_compute_intercept();

    const auto feature_count = data.get_column_count();
    const auto response_count = resp.get_column_count();

    const auto ext_feature_count = feature_count + intp;

    // xtx - Input matrix $X'^T \times X'$ of size P' x P'
    const auto xtx_size = check_mul_overflow(ext_feature_count, ext_feature_count);
    auto xtx_arr = array<Float>::zeros(queue, xtx_size);

    // xty - Input matrix $X'^T \times Y$ of size Ny x P'
    const auto xty_size = check_mul_overflow(response_count, ext_feature_count);
    auto xty_arr = array<Float>::zeros(queue, xty_size);

    // beta - Matrix with regression coefficients of size Ny x (P + 1)
    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(queue, betas_size);

    auto xtx_daal_table =
        interop::convert_to_daal_homogen_table(xtx_arr, ext_feature_count, ext_feature_count);
    auto xty_daal_table =
        interop::convert_to_daal_homogen_table(xty_arr, response_count, ext_feature_count);
    auto betas_daal_table =
        interop::convert_to_daal_homogen_table(betas_arr, response_count, feature_count + 1);

    auto x_daal_table = interop::convert_to_daal_table<Float>(data);
    auto y_daal_table = interop::convert_to_daal_table<Float>(resp);

    {
        const auto status = daal_lr_kernel_t<Float>().compute(*x_daal_table,
                                                              *y_daal_table,
                                                              *xtx_daal_table,
                                                              *xty_daal_table,
                                                              intp);

        interop::status_to_exception(status);
    }

    {
        const auto status = daal_lr_kernel_t<Float>().finalizeCompute(*xtx_daal_table,
                                                                      *xty_daal_table,
                                                                      *xtx_daal_table,
                                                                      *xty_daal_table,
                                                                      *betas_daal_table,
                                                                      intp);

        interop::status_to_exception(status);
    }

    auto betas = homogen_table::wrap(betas_arr, response_count, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(betas);
    const auto model = dal::detail::make_private<model_t>(model_impl);
    auto result = train_result<Task>().set_model(model);

    return result;
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::norm_eq, task::regression>;
template struct train_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend

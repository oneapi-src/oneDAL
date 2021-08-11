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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/low_order_moments/oneapi/low_order_moments_kernel_batch_oneapi.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_gpu;
using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_lom_kernel_t =
    daal_lom::oneapi::internal::LowOrderMomentsBatchKernelOneAPI<Float, daal_lom::defaultDense>;

template <typename Method>
constexpr daal_lom::Method get_daal_method() {
    daal_lom::Method m = daal_lom::defaultDense;

    return m;
}

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const auto daal_data = interop::convert_to_daal_table(queue, data);

    auto daal_parameter = daal_lom::Parameter(get_daal_estimates_to_compute(desc));
    auto daal_input = daal_lom::Input();
    auto daal_result = daal_lom::Result();

    daal_input.set(daal_lom::InputId::data, daal_data);
    interop::status_to_exception(
        daal_result.allocate<Float>(&daal_input, &daal_parameter, get_daal_method<method_t>()));

    interop::status_to_exception(
        daal_lom_kernel_t<Float>().compute(daal_data.get(), &daal_result, &daal_parameter));

    auto result = get_result<Float, task_t>(daal_result);
    result.set_result_options(desc.get_result_options());

    return result;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_gpu<Float, method_t, task_t> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method_t, task_t>;
template struct compute_kernel_gpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend

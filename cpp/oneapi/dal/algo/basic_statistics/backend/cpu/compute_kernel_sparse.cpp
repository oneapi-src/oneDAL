/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <algorithm>

#include "oneapi/dal/algo/basic_statistics/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/low_order_moments/low_order_moments_kernel.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_cpu;
using method_t = method::sparse;
using task_t = task::compute;
using input_t = compute_input<task_t, csr_table>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_lom_batch_kernel_t =
    daal_lom::internal::LowOrderMomentsBatchKernel<Float, daal_lom::fastCSR, Cpu>;

template <typename Float>
result_t call_daal_sparse_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const csr_table& data) {
    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto daal_parameter = daal_lom::Parameter(daal_lom::estimatesAll);
    auto daal_input = daal_lom::Input();
    auto daal_result = daal_lom::Result();

    daal_input.set(daal_lom::InputId::data, daal_data);

    interop::status_to_exception(
        daal_result.allocate<Float>(&daal_input, &daal_parameter, daal_lom::fastCSR));

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_lom_batch_kernel_t>(ctx,
                                                                  daal_data.get(),
                                                                  &daal_result,
                                                                  &daal_parameter));

    auto result =
        get_result<Float, task_t>(desc, daal_result).set_result_options(desc.get_result_options());
    return result;
}

template <typename Float>
struct compute_kernel_cpu<Float, method_t, task_t> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return call_daal_sparse_kernel<Float>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_cpu<float, method_t, task_t>;
template struct compute_kernel_cpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend

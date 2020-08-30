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

#include <daal/src/algorithms/kernel_function/kernel_function_linear_dense_default_kernel.h>

#include "oneapi/dal/algo/linear_kernel/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::linear_kernel::backend {

using dal::backend::context_cpu;

namespace daal_linear_kernel = daal::algorithms::kernel_function::linear;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_linear_kernel_t =
    daal_linear_kernel::internal::KernelImplLinear<daal_linear_kernel::defaultDense, Float, Cpu>;

template <typename Float>
static compute_result call_daal_kernel(const context_cpu& ctx,
                                       const descriptor_base& desc,
                                       const table& x,
                                       const table& y) {
    const int64_t row_count_x = x.get_row_count();
    const int64_t row_count_y = y.get_row_count();
    const int64_t column_count = x.get_column_count();

    auto arr_x = row_accessor<const Float>{ x }.pull();
    auto arr_y = row_accessor<const Float>{ y }.pull();

    auto arr_values = array<Float>::empty(row_count_x * row_count_y);

    const auto daal_x = interop::convert_to_daal_homogen_table(arr_x, row_count_x, column_count);
    const auto daal_y = interop::convert_to_daal_homogen_table(arr_y, row_count_y, column_count);
    const auto daal_values =
        interop::convert_to_daal_homogen_table(arr_values, row_count_x, row_count_y);

    daal_linear_kernel::Parameter daal_parameter(desc.get_scale(), desc.get_shift());

    interop::call_daal_kernel<Float, daal_linear_kernel_t>(ctx,
                                                           daal_x.get(),
                                                           daal_y.get(),
                                                           daal_values.get(),
                                                           &daal_parameter);

    return compute_result().set_values(
        dal::detail::homogen_table_builder{}.reset(arr_values, row_count_x, row_count_y).build());
}

template <typename Float>
static compute_result compute(const context_cpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_x(), input.get_y());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense> {
    compute_result operator()(const context_cpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense>;
template struct compute_kernel_cpu<double, method::dense>;

} // namespace oneapi::dal::linear_kernel::backend

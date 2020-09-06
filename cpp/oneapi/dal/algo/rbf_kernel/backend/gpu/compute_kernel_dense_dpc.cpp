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

#define DAAL_SYCL_INTERFACE
#define DAAL_SYCL_INTERFACE_USM
#define DAAL_SYCL_INTERFACE_REVERSED_RANGE

#include "oneapi/dal/algo/rbf_kernel/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/kernel_function/oneapi/kernel_function_rbf_dense_default_kernel_oneapi.h>

namespace oneapi::dal::rbf_kernel::backend {

using dal::backend::context_gpu;

namespace daal_rbf_kernel = daal::algorithms::kernel_function::rbf;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_rbf_kernel_t =
    daal_rbf_kernel::internal::KernelImplRBFOneAPI<daal_rbf_kernel::defaultDense, Float>;

template <typename Float>
static compute_result call_daal_kernel(const context_gpu& ctx,
                                       const descriptor_base& desc,
                                       const table& x,
                                       const table& y) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count_x = x.get_row_count();
    const int64_t row_count_y = y.get_row_count();
    const int64_t column_count = x.get_column_count();

    auto arr_x = row_accessor<const Float>{ x }.pull(queue);
    auto arr_y = row_accessor<const Float>{ y }.pull(queue);

    auto arr_values = array<Float>::empty(queue, row_count_x * row_count_y);

    const auto daal_x =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_x, row_count_x, column_count);
    const auto daal_y =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_y, row_count_y, column_count);
    const auto daal_values =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_values, row_count_x, row_count_y);

    daal_rbf_kernel::Parameter daal_parameter(desc.get_sigma());
    daal_rbf_kernel_t<Float>().compute(daal_x.get(),
                                       daal_y.get(),
                                       daal_values.get(),
                                       &daal_parameter);

    return compute_result().set_values(
        dal::detail::homogen_table_builder{}.reset(arr_values, row_count_x, row_count_y).build());
}

template <typename Float>
static compute_result compute(const context_gpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_x(), input.get_y());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense> {
    compute_result operator()(const context_gpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense>;
template struct compute_kernel_gpu<double, method::dense>;

} // namespace oneapi::dal::rbf_kernel::backend

#undef DAAL_SYCL_INTERFACE

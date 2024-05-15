/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <daal/src/data_management/finiteness_checker.h>

#include "oneapi/dal/algo/finiteness_checker/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::finiteness_checker::backend {

using dal::backend::context_cpu;
using input_t = compute_input<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_finiteness_checker = daal::algorithms::kernel_function::rbf;


template <typename Float, daal::CpuType Cpu>
using daal_finiteness_checker_t =
    daal::data_management::internal::allValuesAreFinite<Float, Cpu>;

template <typename Float>
static bool call_daal_kernel(const context_cpu& ctx,
                                 const bool desc,
                                 const table& x) {

    const auto daal_x = interop::convert_to_daal_table<Float>(x);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_finiteness_checker_t>(ctx,
                                                            daal_x.get(),
                                                            desc,
                                                            &result));
    return result;
}

template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_x());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense, task::compute> {
    result_t operator()(const context_cpu& ctx,
                        const bool desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_cpu& ctx,
                    bool desc,
                    const table& x,
                    const table& y,
                    bool& res) const {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::finiteness_checker::backend

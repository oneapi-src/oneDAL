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

#include "daal/src/algorithms/covariance/covariance_kernel.h"

#include "oneapi/dal/algo/covariance/backend/cpu/finalize_compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include <iostream>
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_covariance = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_covariance_kernel_t = daal_covariance::internal::
    CovarianceDenseOnlineKernel<Float, daal_covariance::Method::defaultDense, Cpu>;

//TODO:rewrite kernel and add finalize
template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel_finalize_compute(const context_cpu& ctx,
                                                              const descriptor_t& desc,
                                                              const table& data) {
    std::cout << "I call this" << std::endl;
    const std::int64_t component_count = data.get_column_count();

    daal_covariance::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_covariance::covarianceMatrix;

    dal::detail::check_mul_overflow(component_count, component_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto arr_crossproduct = array<Float>::empty(component_count);
    const auto daal_crossproduct =
        interop::convert_to_daal_homogen_table(arr_crossproduct, 1, component_count);
    auto arr_sums = array<Float>::empty(component_count);
    const auto daal_sums = interop::convert_to_daal_homogen_table(arr_sums, 1, component_count);

    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto arr_nobs_matrix = array<Float>::empty(component_count * component_count);
        const auto daal_nobs_matrix = interop::convert_to_daal_homogen_table(arr_nobs_matrix,
                                                                             component_count,
                                                                             component_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       daal_nobs_matrix.get(),
                                                                       daal_crossproduct.get(),
                                                                       daal_sums.get(),
                                                                       &daal_parameter));
        result.set_cov_matrix(
            homogen_table::wrap(arr_nobs_matrix, component_count, component_count));
    }

    return result;
}

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_input<Task>& input) {
    return call_daal_kernel_finalize_compute<Float, Task>(ctx, desc, input.get_data());
}

template <typename Float>
struct finalize_compute_kernel_cpu<Float, method::by_default, task::compute> {
    compute_result<task::compute> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_compute_input<task::compute>& input) const {
        return finalize_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_cpu<float, method::dense, task::compute>;
template struct finalize_compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend

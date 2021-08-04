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

#include "oneapi/dal/algo/covariance/common.hpp"
#include "oneapi/dal/algo/covariance/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "daal/src/algorithms/covariance/covariance_kernel.h"

namespace oneapi::dal::covariance::backend {

using dal::backend::context_cpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_covariance = daal::algorithms::covariance;
namespace daal_covariance_parameter = daal::algorithms::covariance::internal;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_covariance_kernel_t =
    daal_covariance::internal::CovarianceDenseBatchKernel<Float, daal_covariance::Method::defaultDense, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t component_count = data.get_column_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    daal_covariance::Parameter daal_parameter;
    auto result = compute_result<task::compute>{}.set_result_options(desc.get_result_options());

    auto arr_cov_matrix = array<Float>::empty(row_count * component_count);
    auto arr_cor_matrix = array<Float>::empty(row_count * component_count);
    auto arr_means = array<Float>::empty(1 * component_count);

    auto daal_cov_matrix = daal::data_management::NumericTablePtr();
    auto daal_cor_matrix = daal::data_management::NumericTablePtr();
    auto daal_means = daal::data_management::NumericTablePtr();

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        dal::detail::check_mul_overflow(row_count, component_count);
        arr_cov_matrix.reset(row_count * component_count);
        daal_parameter.OutputMatrixType = daal_covariance::covarianceMatrix;
        daal_cov_matrix =
            interop::convert_to_daal_homogen_table(arr_cov_matrix, row_count, component_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(), 
                                                                       daal_cov_matrix.get(),
                                                                       daal_means.get(),
                                                                       &daal_parameter));
        result.set_cov_matrix(
            dal::detail::homogen_table_builder{}.reset(arr_cov_matrix, row_count, component_count).build());
    }

    if (desc.get_result_options().test(result_options::cor_matrix)) {
        dal::detail::check_mul_overflow(row_count, component_count);
        arr_cor_matrix.reset(row_count * component_count);
        daal_parameter.OutputMatrixType = daal_covariance::correlationMatrix;
        daal_cor_matrix =
            interop::convert_to_daal_homogen_table(arr_cor_matrix, row_count, component_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(), 
                                                                       daal_cor_matrix.get(),
                                                                       daal_means.get(),
                                                                       &daal_parameter));
        result.set_cor_matrix(
            dal::detail::homogen_table_builder{}.reset(arr_cor_matrix, row_count, component_count).build());
    }

    if (desc.get_result_options().test(result_options::means)) {
        arr_means.reset(1 * component_count);
        daal_parameter.OutputMatrixType = daal_covariance::OutputMatrixType::covarianceMatrix;
        daal_means =
            interop::convert_to_daal_homogen_table(arr_means, 1, component_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(), 
                                                                       daal_cov_matrix.get(),
                                                                       daal_means.get(),
                                                                       &daal_parameter));
        result.set_means(
            dal::detail::homogen_table_builder{}.reset(arr_means, 1, component_count).build());
    }
    return result;
}


template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense, task::compute> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend

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

#include "daal/src/algorithms/covariance/covariance_kernel.h"

#include "oneapi/dal/algo/covariance/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_covariance = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_covariance_kernel_t = daal_covariance::internal::
    CovarianceDenseBatchKernel<Float, daal_covariance::Method::defaultDense, Cpu>;

template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const table& data) {
    bool is_mean_computed = false;

    const std::int64_t component_count = data.get_column_count();

    daal_covariance::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_covariance::covarianceMatrix;

    daal_covariance::internal::Hyperparameter daal_hyperparameter;
    /// the logic of block size calculation is copied from DAAL,
    /// to be changed to passing the values from the performance model
    std::int64_t blockSize = 140;
    if (ctx.get_enabled_cpu_extensions() == dal::detail::cpu_extension::avx512) {
        const std::int64_t row_count = data.get_row_count();
        if (5000 < row_count && row_count <= 50000) {
            blockSize = 1024;
        }
    }
    interop::status_to_exception(
        daal_hyperparameter.set(daal_covariance::internal::denseUpdateStepBlockSize, blockSize));

    dal::detail::check_mul_overflow(component_count, component_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto arr_means = array<Float>::empty(component_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, component_count);

    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto arr_cov_matrix = array<Float>::empty(component_count * component_count);
        const auto daal_cov_matrix = interop::convert_to_daal_homogen_table(arr_cov_matrix,
                                                                            component_count,
                                                                            component_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       daal_cov_matrix.get(),
                                                                       daal_means.get(),
                                                                       &daal_parameter,
                                                                       &daal_hyperparameter));
        is_mean_computed = true;
        result.set_cov_matrix(
            homogen_table::wrap(arr_cov_matrix, component_count, component_count));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto arr_cor_matrix = array<Float>::empty(component_count * component_count);
        const auto daal_cor_matrix = interop::convert_to_daal_homogen_table(arr_cor_matrix,
                                                                            component_count,
                                                                            component_count);

        daal_parameter.outputMatrixType = daal_covariance::correlationMatrix;

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       daal_cor_matrix.get(),
                                                                       daal_means.get(),
                                                                       &daal_parameter,
                                                                       &daal_hyperparameter));
        is_mean_computed = true;
        result.set_cor_matrix(
            homogen_table::wrap(arr_cor_matrix, component_count, component_count));
    }
    if (desc.get_result_options().test(result_options::means)) {
        if (!is_mean_computed) {
            auto arr_cov_matrix = array<Float>::empty(component_count * component_count);
            const auto daal_cov_matrix = interop::convert_to_daal_homogen_table(arr_cov_matrix,
                                                                                component_count,
                                                                                component_count);
            interop::status_to_exception(
                interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                           daal_data.get(),
                                                                           daal_cov_matrix.get(),
                                                                           daal_means.get(),
                                                                           &daal_parameter,
                                                                           &daal_hyperparameter));
        }
        result.set_means(homogen_table::wrap(arr_means, 1, component_count));
    }
    return result;
}

template <typename Float, typename Task>
static compute_result<Task> compute(const context_cpu& ctx,
                                    const descriptor_t& desc,
                                    const compute_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::by_default, task::compute> {
    compute_result<task::compute> operator()(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const compute_input<task::compute>& input) const {
        return compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend

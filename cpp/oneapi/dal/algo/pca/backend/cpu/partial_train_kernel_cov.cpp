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

#include <daal/src/algorithms/pca/pca_dense_correlation_online_kernel.h>
#include <daal/src/algorithms/covariance/covariance_hyperparameter_impl.h>
#include "daal/src/algorithms/covariance/covariance_kernel.h"

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/partial_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using model_t = model<task::dim_reduction>;
using task_t = task::dim_reduction;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_covariance_kernel_t =
    daal_cov::internal::CovarianceDenseOnlineKernel<Float, daal_cov::Method::defaultDense, Cpu>;

template <typename Float>
static partial_train_result<task_t> call_daal_kernel_partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    const std::int64_t component_count = input.get_data().get_column_count();
    const auto input_ = input.get_prev();
    daal_cov::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_cov::correlationMatrix;

    const auto data = input.get_data();
    ONEDAL_ASSERT(data.has_data());
    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto result = partial_train_result();
    const bool has_nobs_data = input_.get_partial_n_rows().has_data();
    daal_cov::internal::Hyperparameter daal_hyperparameter;
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
        daal_hyperparameter.set(daal_cov::internal::denseUpdateStepBlockSize, blockSize));
    if (has_nobs_data) {
        auto daal_crossproduct =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_crossproduct());
        auto daal_sums = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum());
        auto daal_nobs_matrix =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_n_rows());
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       daal_nobs_matrix.get(),
                                                                       daal_crossproduct.get(),
                                                                       daal_sums.get(),
                                                                       &daal_parameter,
                                                                       &daal_hyperparameter));
        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));
        result.set_partial_n_rows(
            interop::convert_from_daal_homogen_table<Float>(daal_nobs_matrix));
        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct));
    }
    else {
        auto arr_crossproduct = array<Float>::zeros(component_count * component_count);
        auto arr_sums = array<Float>::zeros(component_count);
        auto arr_nobs_matrix = array<Float>::zeros(1 * 1);
        auto daal_crossproduct = interop::convert_to_daal_homogen_table<Float>(arr_crossproduct,
                                                                               component_count,
                                                                               component_count);
        auto daal_sums =
            interop::convert_to_daal_homogen_table<Float>(arr_sums, 1, component_count);
        auto daal_nobs_matrix =
            interop::convert_to_daal_homogen_table<Float>(arr_nobs_matrix, 1, 1);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_covariance_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       daal_nobs_matrix.get(),
                                                                       daal_crossproduct.get(),
                                                                       daal_sums.get(),
                                                                       &daal_parameter,
                                                                       &daal_hyperparameter));

        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));
        result.set_partial_n_rows(
            interop::convert_from_daal_homogen_table<Float>(daal_nobs_matrix));
        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct));
    }

    return result;
}

template <typename Float>
static partial_train_result<task_t> partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    return call_daal_kernel_partial_train<Float>(ctx, desc, input);
}

template <typename Float>
struct partial_train_kernel_cpu<Float, method::cov, task::dim_reduction> {
    partial_train_result<task_t> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_train_input<task::dim_reduction>& input) const {
        return partial_train<Float>(ctx, desc, input);
    }
};

template struct partial_train_kernel_cpu<float, method::cov, task::dim_reduction>;
template struct partial_train_kernel_cpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

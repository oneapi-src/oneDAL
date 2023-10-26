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
#include "oneapi/dal/algo/pca/backend/cpu/finalize_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;
using model_t = model<task::dim_reduction>;

namespace interop = dal::backend::interop;

namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_cor_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::online, Float, Cpu>;

template <typename Float, daal::CpuType Cpu>
using daal_cov_kernel_t =
    daal_cov::internal::CovarianceDenseOnlineKernel<Float, daal_cov::Method::defaultDense, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel_finalize_train(const context_cpu& ctx,
                                                          const descriptor_t& desc,
                                                          const partial_train_result<Task>& input) {
    const std::int64_t component_count =
        get_component_count(desc, input.get_partial_crossproduct());
    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();

    auto result = train_result<task::dim_reduction>{}.set_result_options(desc.get_result_options());

    auto arr_eigvec = array<Float>::empty(column_count * component_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);

    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, component_count, column_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);

    auto arr_means = array<Float>::empty(column_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, column_count);
    daal_cov::internal::Hyperparameter daal_hyperparameter;
    /// the logic of block size calculation is copied from DAAL,
    /// to be changed to passing the values from the performance model
    std::int64_t blockSize = 140;
    if (ctx.get_enabled_cpu_extensions() == dal::detail::cpu_extension::avx512) {
        //const std::int64_t row_count = data.get_row_count();
        //if (5000 < row_count && row_count <= 50000) {
        blockSize = 1024;
        //}
    }
    interop::status_to_exception(
        daal_hyperparameter.set(daal_cov::internal::denseUpdateStepBlockSize, blockSize));
    auto daal_crossproduct =
        interop::convert_to_daal_table<Float>(input.get_partial_crossproduct());
    auto daal_sums = interop::convert_to_daal_table<Float>(input.get_partial_sum());
    const auto daal_nobs = interop::convert_to_daal_table<Float>(input.get_partial_n_rows());

    auto arr_cor_matrix = array<Float>::empty(column_count * column_count);
    const auto daal_cor_matrix =
        interop::convert_to_daal_homogen_table(arr_cor_matrix, column_count, column_count);
    daal_cov::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_cov::correlationMatrix;

    interop::status_to_exception(
        interop::call_daal_kernel_finalize_compute<Float, daal_cov_kernel_t>(
            ctx,
            daal_nobs.get(),
            daal_crossproduct.get(),
            daal_sums.get(),
            daal_cor_matrix.get(),
            daal_means.get(),
            &daal_parameter,
            &daal_hyperparameter));

    const auto data_to_compute = daal_cor_matrix;
    {
        const auto status = dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
            constexpr auto cpu_type = interop::to_daal_cpu_type<decltype(cpu)>::value;
            return daal_pca_cor_kernel_t<Float, cpu_type>().computeCorrelationEigenvalues(
                *data_to_compute,
                *daal_eigenvectors,
                *daal_eigenvalues);
        });

        interop::status_to_exception(status);
    }

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        const auto mdl = model_t{}.set_eigenvectors(
            homogen_table::wrap(arr_eigvec, component_count, column_count));
        result.set_model(mdl);
    }

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(arr_eigval, 1, component_count));
    }

    return result;
}

template <typename Float, typename Task>
static train_result<Task> finalize_train(const context_cpu& ctx,
                                         const descriptor_t& desc,
                                         const partial_train_result<Task>& input) {
    return call_daal_kernel_finalize_train<Float>(ctx, desc, input);
}

template <typename Float>
struct finalize_train_kernel_cpu<Float, method::cov, task::dim_reduction> {
    train_result<task::dim_reduction> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_train_result<task::dim_reduction>& input) const {
        return finalize_train<Float, task::dim_reduction>(ctx, desc, input);
    }
};

template struct finalize_train_kernel_cpu<float, method::cov, task::dim_reduction>;
template struct finalize_train_kernel_cpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

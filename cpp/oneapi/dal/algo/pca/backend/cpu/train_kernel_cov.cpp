/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using model_t = model<task::dim_reduction>;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_cor_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::batch, Float, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = get_component_count(desc, data);

    dal::detail::check_mul_overflow(column_count, component_count);
    auto arr_eigvec = array<Float>::empty(column_count * component_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);
    auto arr_means = array<Float>::empty(1 * column_count);
    auto arr_vars = array<Float>::empty(1 * column_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, component_count, column_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, column_count);
    const auto daal_variances = interop::convert_to_daal_homogen_table(arr_vars, 1, column_count);

    daal_cov::Batch<Float, daal_cov::defaultDense> covariance_alg;
    covariance_alg.input.set(daal_cov::data, daal_data);

    constexpr bool is_correlation = false;
    constexpr std::uint64_t results_to_compute =
        std::uint64_t(daal_pca::mean | daal_pca::variance | daal_pca::eigenvalue);

    interop::status_to_exception(interop::call_daal_kernel<Float, daal_pca_cor_kernel_t>(
        ctx,
        is_correlation,
        desc.get_deterministic(),
        *daal_data,
        &covariance_alg,
        static_cast<DAAL_UINT64>(results_to_compute),
        *daal_eigenvectors,
        *daal_eigenvalues,
        *daal_means,
        *daal_variances));

    // clang-format off
    const auto mdl = model_t{}
        .set_eigenvectors(
            dal::detail::homogen_table_builder{}
                .reset(arr_eigvec, component_count, column_count)
                .build()
        );

    return result_t{}
        .set_model(mdl)
        .set_eigenvalues(
            dal::detail::homogen_table_builder{}
                .reset(arr_eigval, 1, component_count)
                .build()
        )
        .set_variances(
            dal::detail::homogen_table_builder{}
                .reset(arr_vars, 1, column_count)
                .build()
        )
        .set_means(
            dal::detail::homogen_table_builder{}
                .reset(arr_means, 1, column_count)
                .build()
        );
    // clang-format on
}

template <typename Float>
static result_t train(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct train_kernel_cpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::cov, task::dim_reduction>;
template struct train_kernel_cpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

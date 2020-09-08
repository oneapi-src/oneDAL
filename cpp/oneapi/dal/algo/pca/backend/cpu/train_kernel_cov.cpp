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

#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>

#include "oneapi/dal/algo/pca/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using std::int64_t;
using dal::backend::context_cpu;

namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_cor_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::batch, Float, Cpu>;

template <typename Float>
static train_result call_daal_kernel(const context_cpu& ctx,
                                     const descriptor_base& desc,
                                     const table& data) {
    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();
    const int64_t component_count = desc.get_component_count();

    auto arr_data = row_accessor<const Float>{ data }.pull();
    auto arr_eigvec = array<Float>::empty(column_count * component_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);
    auto arr_means = array<Float>::empty(1 * component_count);
    auto arr_vars = array<Float>::empty(1 * component_count);

    // TODO: read-only access performed with deep copy of data since daal numeric tables are mutable.
    // Need to create special immutable homogen table on daal interop side

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, column_count, component_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, component_count);
    const auto daal_variances =
        interop::convert_to_daal_homogen_table(arr_vars, 1, component_count);

    daal_cov::Batch<Float, daal_cov::defaultDense> covariance_alg;
    covariance_alg.input.set(daal_cov::data, daal_data);

    constexpr bool is_correlation = false;
    constexpr uint64_t results_to_compute =
        int64_t(daal_pca::mean | daal_pca::variance | daal_pca::eigenvalue);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_pca_cor_kernel_t>(ctx,
                                                                is_correlation,
                                                                desc.get_is_deterministic(),
                                                                *daal_data,
                                                                &covariance_alg,
                                                                results_to_compute,
                                                                *daal_eigenvectors,
                                                                *daal_eigenvalues,
                                                                *daal_means,
                                                                *daal_variances));

    return train_result()
        .set_model(model().set_eigenvectors(dal::detail::homogen_table_builder{}
                                                .reset(arr_eigvec, column_count, component_count)
                                                .build()))
        .set_eigenvalues(
            dal::detail::homogen_table_builder{}.reset(arr_eigval, 1, component_count).build());
}

template <typename Float>
static train_result train(const context_cpu& ctx,
                          const descriptor_base& desc,
                          const train_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct train_kernel_cpu<Float, method::cov> {
    train_result operator()(const context_cpu& ctx,
                            const descriptor_base& desc,
                            const train_input& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::cov>;
template struct train_kernel_cpu<double, method::cov>;

} // namespace oneapi::dal::pca::backend

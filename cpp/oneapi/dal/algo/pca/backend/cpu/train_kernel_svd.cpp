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

#include <daal/src/algorithms/pca/pca_dense_svd_batch_kernel.h>
#include <daal/include/algorithms/normalization/zscore_types.h>

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
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_svd_kernel_t = daal_pca::internal::
    PCASVDBatchKernel<Float, daal_pca::interface3::BatchParameter<Float, daal_pca::svdDense>, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = get_component_count(desc, data);

    auto arr_data = row_accessor<const Float>{ data }.pull();

    dal::detail::check_mul_overflow(column_count, component_count);
    auto arr_eigvec = array<Float>::empty(column_count * component_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);
    auto arr_means = array<Float>::empty(1 * column_count);
    auto arr_vars = array<Float>::empty(1 * column_count);

    // TODO: read-only access performed with deep copy of data since daal numeric tables are mutable.
    // Need to create special immutable homogen table on daal interop side

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, column_count, component_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, column_count);
    const auto daal_variances = interop::convert_to_daal_homogen_table(arr_vars, 1, column_count);

    daal_pca::internal::InputDataType dtype = daal_pca::internal::nonNormalizedDataset;

    daal_pca::interface3::BatchParameter<Float, daal_pca::svdDense> parameter;

    auto normalizationAlgorithm = parameter.normalization;
    normalizationAlgorithm->input.set(daal::algorithms::normalization::zscore::data, daal_data);
    auto algParameter = &(normalizationAlgorithm->parameter());
    algParameter->resultsToCompute |= daal::algorithms::normalization::zscore::mean;
    algParameter->resultsToCompute |= daal::algorithms::normalization::zscore::variance;

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_pca_svd_kernel_t>(ctx,
                                                                dtype,
                                                                *daal_data.get(),
                                                                &parameter,
                                                                *daal_eigenvalues.get(),
                                                                *daal_eigenvectors.get(),
                                                                *daal_means.get(),
                                                                *daal_variances.get()));

    // clang-format off
    const auto mdl = model_t{}
        .set_eigenvectors(
            dal::detail::homogen_table_builder{}
                .reset(arr_eigvec, component_count, column_count)
                .build()
        );

    return result_t()
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
struct train_kernel_cpu<Float, method::svd, task::dim_reduction> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::svd, task::dim_reduction>;
template struct train_kernel_cpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

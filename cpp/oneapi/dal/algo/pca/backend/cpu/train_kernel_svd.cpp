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
using task_t = task::dim_reduction;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_pca = daal::algorithms::pca;
namespace daal_zscore = daal::algorithms::normalization::zscore;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_svd_kernel_t = daal_pca::internal::
    PCASVDBatchKernel<Float, daal_pca::BatchParameter<Float, daal_pca::svdDense>, Cpu>;

template <typename Float>
inline auto get_normalization_algorithm() {
    using normalization_alg_t = daal_zscore::Batch<Float, daal_zscore::defaultDense>;
    return daal::services::SharedPtr<normalization_alg_t>{ new normalization_alg_t{} };
}

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());
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

    daal_pca::internal::InputDataType dtype = daal_pca::internal::nonNormalizedDataset;
    //TODO: investigate opportunity to change default behavior
    if (desc.is_scaled() == true && desc.is_mean_centered() == true) {
        dtype = daal_pca::internal::normalizedDataset;
    }

    auto norm_alg = get_normalization_algorithm<Float>();
    norm_alg->input.set(daal_zscore::data, daal_data);
    norm_alg->parameter().resultsToCompute |= daal_zscore::mean;
    norm_alg->parameter().resultsToCompute |= daal_zscore::variance;
    daal_pca::BatchParameter<Float, daal_pca::svdDense> parameter;
    norm_alg->parameter().doScale = true;
    parameter.doScale = true;
    if (desc.do_mean_centering() == true && desc.do_scale() == false) {
        norm_alg->parameter().doScale = false;
        parameter.doScale = false;
    }
    parameter.isDeterministic = desc.get_deterministic();
    parameter.normalization = norm_alg;
    parameter.resultsToCompute =
        std::uint64_t(daal_pca::mean | daal_pca::variance | daal_pca::eigenvalue);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_pca_svd_kernel_t>(ctx,
                                                                dtype,
                                                                *daal_data.get(),
                                                                &parameter,
                                                                *daal_eigenvalues.get(),
                                                                *daal_eigenvectors.get(),
                                                                *daal_means.get(),
                                                                *daal_variances.get()));
    model_t model;
    if (desc.get_result_options().test(result_options::eigenvectors)) {
        model.set_eigenvectors(homogen_table::wrap(arr_eigvec, component_count, column_count));
    }

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(arr_eigval, 1, component_count));
        result.set_singular_values(homogen_table::wrap(arr_eigval, 1, component_count));
    }
    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(homogen_table::wrap(arr_vars, 1, column_count));
    }
    if (desc.get_result_options().test(result_options::means)) {
        result.set_means(homogen_table::wrap(arr_means, 1, column_count));
    }
    result.set_model(model);
    return result;
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

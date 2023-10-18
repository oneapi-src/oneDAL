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

#include <daal/src/algorithms/pca/pca_dense_correlation_online_kernel.h>
#include <daal/src/algorithms/covariance/covariance_hyperparameter_impl.h>

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/partial_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include <iostream>

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
using daal_pca_cor_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::online, Float, Cpu>;

template <typename Float>
static partial_train_result<task_t> call_daal_kernel_partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    //ONEDAL_ASSERT(data.has_data());
    auto result = partial_train_result();
    auto daal_input = daal_pca::Input();
    auto daal_partial = daal_pca::PartialResult<daal::algorithms::pca::correlationDense>();

    const auto input_ = input.get_prev();
    const auto data = input.get_data();
    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const std::int64_t component_count = input.get_data().get_column_count();

    auto daal_parameter = daal_pca::OnlineParameter<Float, daal::algorithms::pca::correlationDense>(
        daal::services::SharedPtr<daal_cov::Online<Float, daal_cov::defaultDense>>(
            new daal_cov::Online<Float, daal_cov::defaultDense>()));

    const bool has_nobs_data = input_.get_partial_n_rows().has_data();

    if (has_nobs_data) {
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_n_rows());
        daal_partial.set(daal_pca::PartialCorrelationResultId::nObservationsCorrelation, daal_nobs);
        auto daal_crossproduct =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_crossproduct());
        daal_partial.set(daal_pca::PartialCorrelationResultId::crossProductCorrelation,
                         daal_crossproduct);
        auto daal_sums = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum());
        daal_partial.set(daal_pca::PartialCorrelationResultId::sumCorrelation, daal_sums);

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_cor_kernel_t>(ctx,
                                                                    daal_data,
                                                                    &daal_partial,
                                                                    &daal_parameter));
        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));
        result.set_partial_n_rows(interop::convert_from_daal_homogen_table<Float>(daal_nobs));
        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct));

        return result;
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
        auto daal_nobs = interop::convert_to_daal_homogen_table<Float>(arr_nobs_matrix, 1, 1);

        daal_partial.set(daal_pca::PartialCorrelationResultId::nObservationsCorrelation, daal_nobs);

        daal_partial.set(daal_pca::PartialCorrelationResultId::crossProductCorrelation,
                         daal_crossproduct);

        daal_partial.set(daal_pca::PartialCorrelationResultId::sumCorrelation, daal_sums);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_cor_kernel_t>(ctx,
                                                                    daal_data,
                                                                    &daal_partial,
                                                                    &daal_parameter));

        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));
        result.set_partial_n_rows(interop::convert_from_daal_homogen_table<Float>(daal_nobs));
        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct));

        return result;
    }
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

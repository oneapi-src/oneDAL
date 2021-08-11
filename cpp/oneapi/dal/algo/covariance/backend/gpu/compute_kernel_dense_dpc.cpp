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

#include "src/algorithms/kernel.h"
#include "daal/src/algorithms/covariance/oneapi/covariance_kernel_oneapi.h"

#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_covariance = daal::algorithms::covariance;
namespace daal_covariance_parameter = daal::algorithms::covariance::oneapi::internal;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_covariance_kernel_t = daal_covariance::oneapi::internal::
    CovarianceDenseBatchKernelOneAPI<Float, daal_covariance::Method::defaultDense>;

template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t component_count = data.get_column_count();

    daal_covariance::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_covariance::covarianceMatrix;

    auto arr_cov_matrix =
        array<Float>::empty(queue, component_count * component_count, sycl::usm::alloc::device);
    auto arr_cor_matrix =
        array<Float>::empty(queue, component_count * component_count, sycl::usm::alloc::device);
    auto arr_means = array<Float>::empty(queue, component_count, sycl::usm::alloc::device);

    const auto daal_data = interop::convert_to_daal_table(queue, data);

    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        const auto daal_means =
            interop::convert_to_daal_table(queue, arr_means, 1, component_count);

        const auto daal_cov_matrix =
            interop::convert_to_daal_table(queue, arr_cov_matrix, component_count, component_count);

        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_cov_matrix.get(),
                                                      daal_means.get(),
                                                      &daal_parameter));
        result.set_cov_matrix(
            homogen_table::wrap(arr_cov_matrix, component_count, component_count));
        result.set_means(homogen_table::wrap(arr_means, 1, component_count));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        const auto daal_means =
            interop::convert_to_daal_table(queue, arr_means, 1, component_count);

        const auto daal_cor_matrix =
            interop::convert_to_daal_table(queue, arr_cor_matrix, component_count, component_count);
        daal_parameter.outputMatrixType = daal_covariance::correlationMatrix;

        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_cor_matrix.get(),
                                                      daal_means.get(),
                                                      &daal_parameter));
        result.set_cor_matrix(
            homogen_table::wrap(arr_cor_matrix, component_count, component_count));
        result.set_means(homogen_table::wrap(arr_means, 1, component_count));
    }
    if (desc.get_result_options().test(result_options::means) && !result.get_means().has_data()) {
        const auto daal_means =
            interop::convert_to_daal_table(queue, arr_means, 1, component_count);

        const auto daal_cov_matrix =
            interop::convert_to_daal_table(queue, arr_cov_matrix, component_count, component_count);

        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_cov_matrix.get(),
                                                      daal_means.get(),
                                                      &daal_parameter));
        result.set_means(homogen_table::wrap(arr_means, 1, component_count));
    }
    return result;
}

template <typename Float, typename Task>
static compute_result<Task> compute(const context_gpu& ctx,
                                    const descriptor_t& desc,
                                    const compute_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::by_default, task::compute> {
    compute_result<task::compute> operator()(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const compute_input<task::compute>& input) const {
        return compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend

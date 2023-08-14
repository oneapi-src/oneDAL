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

#include "daal/src/algorithms/covariance/oneapi/covariance_kernel_oneapi.h"

#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_covariance = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_covariance_kernel_t = daal_covariance::oneapi::internal::
    CovarianceDenseOnlineKernelOneAPI<Float, daal_covariance::Method::defaultDense>;

template <typename Float, typename Task>
static partial_compute_input<Task> call_daal_kernel_partial_compute(
    const context_gpu& ctx,
    const descriptor_t& desc,
    const partial_compute_input<Task>& input) {
    //bool is_mean_computed = false;

    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t component_count = input.get_data().get_column_count();

    daal_covariance::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_covariance::covarianceMatrix;

    dal::detail::check_mul_overflow(component_count, component_count);
    const auto data = input.get_data();
    const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);

    auto result = partial_compute_input(input);

    const bool has_nobs_data = input.get_nobs_table().has_data();

    if (has_nobs_data) {
        // auto daal_crossproduct =
        //     interop::copy_to_daal_homogen_table(input.get_crossproduct_matrix());
        // auto daal_sums = interop::copy_to_daal_homogen_table(input.get_sums());
        // auto daal_nobs_matrix = interop::copy_to_daal_homogen_table(input.get_nobs_table());
        // interop::status_to_exception(
        //     daal_covariance_kernel_t<Float>().compute(daal_data.get(),
        //                                               daal_nobs_matrix.get(),
        //                                               daal_crossproduct.get(),
        //                                               daal_sums.get(),
        //                                               &daal_parameter));
        // auto partial_result_sums_arr = interop::convert_from_daal_homogen_table<Float>(daal_sums);
        // auto partial_result_nobs_arr =
        //     interop::convert_from_daal_homogen_table<int>(daal_nobs_matrix);
        // auto partial_result_crossproduct_arr =
        //     interop::convert_from_daal_homogen_table<Float>(daal_crossproduct);
        // result.set_sums(partial_result_sums_arr);
        // result.set_nobs_table(partial_result_nobs_arr);
        // result.set_crossproduct_matrix(partial_result_crossproduct_arr);
    }
    else {
        auto arr_crossproduct =
            array<Float>::empty(queue, component_count * component_count, sycl::usm::alloc::device);
        auto arr_sums = array<Float>::empty(queue, component_count, sycl::usm::alloc::device);
        auto arr_nobs_matrix = array<int>::empty(queue, 1 * 1, sycl::usm::alloc::device);
        const auto daal_crossproduct = interop::convert_to_daal_homogen_table(arr_crossproduct,
                                                                              component_count,
                                                                              component_count);
        const auto daal_sums = interop::convert_to_daal_homogen_table(arr_sums, 1, component_count);
        const auto daal_nobs_matrix = interop::convert_to_daal_homogen_table(arr_nobs_matrix, 1, 1);
        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_nobs_matrix.get(),
                                                      daal_crossproduct.get(),
                                                      daal_sums.get(),
                                                      &daal_parameter));

        result.set_sums(homogen_table::wrap(arr_sums, component_count, 1));
        result.set_nobs_table(homogen_table::wrap(arr_nobs_matrix, 1, 1));
        result.set_crossproduct_matrix(
            homogen_table::wrap(arr_crossproduct, component_count, component_count));
    }

    return result;
}

template <typename Float, typename Task>
static partial_compute_input<Task> partial_compute(const context_gpu& ctx,
                                                   const descriptor_t& desc,
                                                   const partial_compute_input<Task>& input) {
    return call_daal_kernel_partial_compute<Float, Task>(ctx, desc, input);
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method::by_default, task::compute> {
    partial_compute_input<task::compute> operator()(
        const context_gpu& ctx,
        const descriptor_t& desc,
        const partial_compute_input<task::compute>& input) const {
        return partial_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method::dense, task::compute>;
// template struct partial_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend
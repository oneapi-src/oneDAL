/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "daal/src/algorithms/spectral_embedding/spectral_embedding_kernel.h"

#include "oneapi/dal/algo/spectral_embedding/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include <iostream>

namespace oneapi::dal::spectral_embedding::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace sp_emb = daal::algorithms::spectral_embedding;

template <typename Float, daal::CpuType Cpu>
using daal_sp_emb_kernel_t =
    sp_emb::internal::SpectralEmbeddingKernel<Float, sp_emb::Method::defaultDense, Cpu>;

using parameter_t = sp_emb::internal::KernelParameter;


namespace interop = oneapi::dal::backend::interop;

template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const table& data) {
    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    const std::int64_t p = data.get_column_count();
    const std::int64_t n = data.get_row_count();

    std::cout << "inside oneDAL kernel: " << n << " " << p << std::endl; 

    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());




    if (result.get_result_options().test(result_options::embedding)) {
        
        daal::services::SharedPtr<NumericTable> daal_input, daal_output;
        array<Float> arr_output;
        arr_output = array<Float>::empty(n * n);
        daal_output = interop::convert_to_daal_homogen_table(arr_output, n, n);
        parameter_t daal_param;

        daal_param.num_emb = desc.get_embedding_dim();
        daal_param.p = desc.get_num_neighbors();
        interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_sp_emb_kernel_t>(ctx,
                                                                daal_data.get(),
                                                                daal_output.get(),
                                                                daal_param));
        
        result.set_embedding(homogen_table::wrap(arr_output, n, n));
    }



    /*
    parameter_t daal_parameter(n, // number of terms
                               daal::data_management::NumericTablePtr(),
                               0);

    const auto obj_impl = detail::get_objective_impl(desc);
    daal_parameter.penaltyL1 = obj_impl->get_l1_regularization_coefficient();
    daal_parameter.penaltyL2 = obj_impl->get_l2_regularization_coefficient();
    daal_parameter.interceptFlag = obj_impl->get_intercept_flag();

    array<Float> arr_val, arr_grad, arr_hess;

    daal::services::SharedPtr<NumericTable> daal_val;
    daal::services::SharedPtr<NumericTable> daal_gradient;
    daal::services::SharedPtr<NumericTable> daal_hessian;

    if (result.get_result_options().test(result_options::value)) {
        arr_val = array<Float>::empty(1);
        daal_val = interop::convert_to_daal_homogen_table(arr_val, 1, 1);
        daal_parameter.resultsToCompute |= daal_obj_fun::value;
    }

    if (result.get_result_options().test(result_options::gradient)) {
        arr_grad = array<Float>::empty(p + 1);
        daal_gradient = interop::convert_to_daal_homogen_table(arr_grad, p + 1, 1);
        daal_parameter.resultsToCompute |= daal_obj_fun::gradient;
    }

    if (result.get_result_options().test(result_options::hessian)) {
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, p + 1, p + 1);
        arr_hess = array<Float>::empty((p + 1) * (p + 1));
        daal_hessian = interop::convert_to_daal_homogen_table(arr_hess, p + 1, p + 1);
        daal_parameter.resultsToCompute |= daal_obj_fun::hessian;
    }

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_logloss_kernel_t>(ctx,
                                                                daal_data.get(),
                                                                daal_resp.get(),
                                                                daal_params.get(),
                                                                daal_val.get(),
                                                                daal_hessian.get(),
                                                                daal_gradient.get(),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                &daal_parameter));

    if (result.get_result_options().test(result_options::value)) {
        result.set_value(homogen_table::wrap(arr_val, 1, 1));
    }

    if (result.get_result_options().test(result_options::gradient)) {
        result.set_gradient(homogen_table::wrap(arr_grad, p + 1, 1));
    }

    if (result.get_result_options().test(result_options::hessian)) {
        result.set_hessian(homogen_table::wrap(arr_hess, p + 1, p + 1));
    }
    */
    return result;
}

template <typename Float, typename Task>
static compute_result<Task> compute(const context_cpu& ctx,
                                    const descriptor_t& desc,
                                    const compute_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx,
                                         desc,
                                         input.get_data());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::by_default, task::compute> {
    compute_result<task::compute> operator()(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const compute_input<task::compute>& input) const {
        return compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense_batch, task::compute>;
template struct compute_kernel_cpu<double, method::dense_batch, task::compute>;

} // namespace oneapi::dal::spectral_embedding::backend

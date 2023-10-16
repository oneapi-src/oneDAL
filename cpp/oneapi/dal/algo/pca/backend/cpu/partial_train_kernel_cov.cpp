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

//#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>
//#include <daal/src/algorithms/covariance/covariance_hyperparameter_impl.h>

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

// namespace daal_pca = daal::algorithms::pca;
// // daal_cov = daal::algorithms::covariance;
// namespace interop = dal::backend::interop;

template <typename Float>
static partial_train_result<task_t> call_daal_kernel_partial_train(const context_cpu& ctx,
                                                                   const descriptor_t& desc,
                                                                   const table& data) {
    //const std::int64_t column_count = data.get_column_count();
    auto result = partial_train_result();
    std::cout << "partial train" << std::endl;
    return result;
}

template <typename Float>
static partial_train_result<task_t> partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    return call_daal_kernel_partial_train<Float>(ctx, desc, input.get_data());
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

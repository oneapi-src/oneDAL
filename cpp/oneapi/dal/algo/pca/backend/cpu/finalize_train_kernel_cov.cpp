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

//#include "daal/src/algorithms/pca/pca_kernel.h"

#include "oneapi/dal/algo/pca/backend/cpu/finalize_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include <iostream>
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

// namespace daal_pca = daal::algorithms::pca;
namespace interop = dal::backend::interop;

// template <typename Float, daal::CpuType Cpu>
// using daal_pca_kernel_t = daal_pca::internal::
//     pcaDenseOnlineKernel<Float, daal_pca::Method::defaultDense, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel_finalize(const context_cpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_train_result<Task>& input) {
    //const std::int64_t component_count = input.get_partial_crossproduct().get_column_count();

    auto result = train_result<Task>{};
    std::cout << "finalize train" << std::endl;
    return result;
}

template <typename Float, typename Task>
static train_result<Task> finalize_train(const context_cpu& ctx,
                                         const descriptor_t& desc,
                                         const partial_train_result<Task>& input) {
    return call_daal_kernel_finalize<Float, Task>(ctx, desc, input);
}

template <typename Float>
struct finalize_train_kernel_cpu<Float, method::by_default, task::dim_reduction> {
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

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

#include <daal/src/algorithms/linear_regression/linear_regression_train_kernel.h>
#include <daal/src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include <iostream>
#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/finalize_train_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace be = dal::backend;
namespace pr = be::primitives;
namespace interop = dal::backend::interop;
namespace daal_lr = daal::algorithms::linear_regression;

using daal_hyperparameters_t = daal_lr::internal::Hyperparameter;

constexpr auto daal_method = daal_lr::training::normEqDense;

template <typename Float, daal::CpuType Cpu>
using online_kernel_t = daal_lr::training::internal::OnlineKernel<Float, daal_method, Cpu>;

template <typename Float>
static train_result<task::regression> call_daal_kernel(
    const context_cpu& ctx,
    const detail::descriptor_base<task::regression>& desc,
    const partial_train_result<task::regression>& input) {
    using dal::detail::check_mul_overflow;

    std::cout << "Calling finalize train" << std::endl;
    auto result = train_result<task::regression>();

    return result;
}

template <typename Float>
static train_result<task::regression> train(const context_cpu& ctx,
                                            const detail::descriptor_base<task::regression>& desc,
                                            const partial_train_result<task::regression>& input) {
    return call_daal_kernel<Float>(ctx, desc, input);
}

template <typename Float>
struct finalize_train_kernel_cpu<Float, method::norm_eq, task::regression> {
    train_result<task::regression> operator()(
        const context_cpu& ctx,
        const detail::descriptor_base<task::regression>& desc,
        const partial_train_result<task::regression>& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct finalize_train_kernel_cpu<float, method::norm_eq, task::regression>;
template struct finalize_train_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend

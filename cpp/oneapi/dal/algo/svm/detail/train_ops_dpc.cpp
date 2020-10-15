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

#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/detail/train_ops.hpp"
#include "oneapi/dal/backend/dispatcher_dpc.hpp"

namespace oneapi::dal::svm::detail {
using oneapi::dal::detail::data_parallel_policy;

template <typename Float, typename Task, typename Method>
struct train_ops_dispatcher<data_parallel_policy, Float, Task, Method> {
    train_result operator()(const data_parallel_policy& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        using kernel_dispatcher_t =
            dal::backend::kernel_dispatcher<backend::train_kernel_cpu<Float, Task, Method>,
                                            backend::train_kernel_gpu<Float, Task, Method>>;
        return kernel_dispatcher_t{}(ctx, params, input);
    }
};

#define INSTANTIATE(F, T, M) \
    template struct ONEAPI_DAL_EXPORT train_ops_dispatcher<data_parallel_policy, F, T, M>;

INSTANTIATE(float, task::classification, method::smo)
INSTANTIATE(float, task::classification, method::thunder)
INSTANTIATE(double, task::classification, method::smo)
INSTANTIATE(double, task::classification, method::thunder)

} // namespace oneapi::dal::svm::detail

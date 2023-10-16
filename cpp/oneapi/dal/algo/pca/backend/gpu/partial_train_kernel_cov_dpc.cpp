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

#include "oneapi/dal/algo/pca/backend/gpu/partial_train_kernel.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::dim_reduction;
using input_t = partial_train_input<task_t>;
using result_t = partial_train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, typename Task>
static partial_train_result<Task> partial_train(const context_gpu& ctx,
                                                const descriptor_t& desc,
                                                const partial_train_input<Task>& input) {
    auto result = partial_train_result();

    return result;
}

template <typename Float>
struct partial_train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_train<Float, task::dim_reduction>(ctx, desc, input);
    }
};

template struct partial_train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct partial_train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend

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

#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::covariance::backend {

namespace pr = oneapi::dal::backend::primitives;

using method_t = method::dense;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
static result_t partial_compute(const bk::context_gpu& ctx,
                                const descriptor_t& desc,
                                const input_t& input) {
    return partial_compute_kernel_dense_impl<Float>(ctx)(desc, input);
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method_t, task_t> {
    result_t operator()(const bk::context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method_t, task_t>;
template struct partial_compute_kernel_gpu<double, method_t, task_t>;

} // namespace oneapi::dal::covariance::backend

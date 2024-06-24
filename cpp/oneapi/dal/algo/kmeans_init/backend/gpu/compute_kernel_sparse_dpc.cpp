/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using dal::backend::context_gpu;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace ki = oneapi::dal::kmeans_init;

using task_t = task::init;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = ki::detail::descriptor_base<task_t>;

template <typename Float, typename Method>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    constexpr bool is_random_csr = std::is_same_v<Method, method::random_csr>;
    constexpr bool is_plus_plus_csr = std::is_same_v<Method, method::plus_plus_csr>;
    constexpr bool is_parallel_plus_csr = std::is_same_v<Method, method::parallel_plus_csr>;
    using msg = dal::detail::error_messages;
    if constexpr (is_random_csr || is_plus_plus_csr || is_parallel_plus_csr) {
        throw unimplemented(msg::kmeans_init_csr_methods_are_not_implemented_for_gpu());
    }
}

template <typename Float, typename Method>
struct compute_kernel_gpu<Float, Method, task::init> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float, Method>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::random_csr, task::init>;
template struct compute_kernel_gpu<double, method::random_csr, task::init>;
template struct compute_kernel_gpu<float, method::plus_plus_csr, task::init>;
template struct compute_kernel_gpu<double, method::plus_plus_csr, task::init>;
template struct compute_kernel_gpu<float, method::parallel_plus_csr, task::init>;
template struct compute_kernel_gpu<double, method::parallel_plus_csr, task::init>;

} // namespace oneapi::dal::kmeans_init::backend

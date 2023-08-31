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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel.hpp"

// #include "oneapi/dal/backend/primitives/lapack.hpp"
// #include "oneapi/dal/backend/primitives/reduction.hpp"
// #include "oneapi/dal/backend/primitives/stat.hpp"
// #include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
//namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task::compute>;

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    //auto& q = ctx.get_queue();

    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    //const auto nobs_host = pr::table2ndarray<Float>(q, input.get_nobs());
    //auto rows_count_global = nobs_host.get_data()[0];

    return result;
}

template <typename Float>
struct finalize_compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return finalize_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_gpu<float, method::dense, task::compute>;
template struct finalize_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::basic_statistics::backend

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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/covariance/common.hpp"
#include "oneapi/dal/algo/covariance/compute_types.hpp"

#include "oneapi/dal/algo/covariance/parameters/gpu/compute_parameters.hpp"

namespace oneapi::dal::covariance::parameters {

using dal::backend::context_gpu;
/*
template <typename Float>
std::int64_t propose_block_size(const sycl::queue& q) {
    return 0l;
}
*/
template <typename Float, typename Task>
struct compute_parameters_gpu<Float, method::norm_eq, Task> {
    using params_t = detail::compute_parameters<Task>;
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const compute_input<Task>& input) const {
        const auto& queue = ctx.get_queue();

        /* const auto& x_train = input.get_data();
        const auto& y_train = input.get_responses();

        const auto f_count = x_train.get_column_count();
        const auto r_count = y_train.get_column_count();

        const auto block = propose_block_size<Float>(queue, f_count, r_count);

        return params_t{}.set_gpu_macro_block(block); */
        return params_t{};
    }
};

template struct ONEDAL_EXPORT compute_parameters_gpu<float, method::dense, task::covariance>;
template struct ONEDAL_EXPORT compute_parameters_gpu<double, method::dense, task::covariance>;

} // namespace oneapi::dal::covariance::parameters

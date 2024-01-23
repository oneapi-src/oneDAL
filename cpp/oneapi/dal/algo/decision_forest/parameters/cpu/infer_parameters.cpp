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

#include "oneapi/dal/algo/decision_forest/parameters/cpu/infer_parameters.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_cpu;

std::int64_t propose_block_size(const context_cpu& ctx) {
    std::int64_t block_size = 22l;
    if (ctx.get_enabled_cpu_extensions() == dal::detail::cpu_extension::avx512) {
        /// Here if AVX512 extensions are available on CPU
        block_size = 32l;
    }
    return block_size;
}

constexpr std::int64_t propose_min_trees_for_threading() {
    return 100l;
}

constexpr std::int64_t propose_min_number_of_rows_for_vect_seq_compute() {
    return 32l;
}

constexpr double propose_scale_factor_for_vect_seq_compute() {
    return 0.3f;
}

// TODO: Is it correct to use method::by_default here?
template <typename Float, typename Task>
struct infer_parameters_cpu<Float, method::by_default, Task> {
    using params_t = detail::infer_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const infer_input<Task>& input) const {
        const auto block = propose_block_size(ctx);
        const auto trees = propose_min_trees_for_threading();
        const auto seq_nrows = propose_min_number_of_rows_for_vect_seq_compute();
        const auto seq_trees = propose_scale_factor_for_vect_seq_compute();

        return params_t{}
            .set_block_size(block)
            .set_min_trees_for_threading(trees)
            .set_min_number_of_rows_for_vect_seq_compute(seq_nrows)
            .set_scale_factor_for_vect_parallel_compute(seq_trees);
    }
};

template struct ONEDAL_EXPORT infer_parameters_cpu<float, method::dense, task::by_default>;
template struct ONEDAL_EXPORT infer_parameters_cpu<double, method::dense, task::by_default>;

} // namespace oneapi::dal::decision_forest::parameters

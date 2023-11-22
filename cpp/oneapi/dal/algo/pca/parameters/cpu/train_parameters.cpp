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

#include <algorithm>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/pca/common.hpp"
#include "oneapi/dal/algo/pca/train_types.hpp"

#include "oneapi/dal/algo/pca/parameters/cpu/train_parameters.hpp"

namespace oneapi::dal::pca::parameters {

using dal::backend::context_cpu;

/// Proposes the number of rows in the data block used in variance-pca matrix computations on CPU.
///
/// @tparam Float   The type of elements that is used in computations in pca algorithm.
///                 The :literal:`Float` type should be at least :expr:`float` or :expr:`double`.
///
/// @param[in] ctx       Context that stores the information about the available CPU extensions
///                      and available data communication mechanisms, parallel or distributed.
/// @param[in] row_count Number of rows in the input dataset.
///
/// @return Number of rows in the data block used in variance-pca matrix computations on CPU.
template <typename Float>
std::int64_t propose_block_size(const context_cpu& ctx, const std::int64_t row_count) {
    /// The constants are defined as the values that show the best performance results
    /// in the series of performance measurements with the varying block sizes and dataset sizes.
    std::int64_t block_size = 140l;
    if (ctx.get_enabled_cpu_extensions() == dal::detail::cpu_extension::avx512) {
        /// Here if AVX512 extensions are available on CPU
        if (5000l < row_count && row_count <= 50000l) {
            block_size = 1024l;
        }
    }
    return block_size;
}

template <typename Float, typename Task>
struct train_parameters_cpu<Float, method::cov, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        const auto& x = input.get_data();

        const auto row_count = x.get_row_count();

        const auto block = propose_block_size<Float>(ctx, row_count);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_input<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_result<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
};

template <typename Float, typename Task>
struct train_parameters_cpu<Float, method::svd, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        const auto& x = input.get_data();

        const auto row_count = x.get_row_count();

        const auto block = propose_block_size<Float>(ctx, row_count);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_input<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_result<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
};

template <typename Float, typename Task>
struct train_parameters_cpu<Float, method::precomputed, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        const auto& x = input.get_data();

        const auto row_count = x.get_row_count();

        const auto block = propose_block_size<Float>(ctx, row_count);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_input<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_result<Task>& input) const {
        const auto block = propose_block_size<Float>(ctx, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
};

template struct ONEDAL_EXPORT train_parameters_cpu<float, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT
    train_parameters_cpu<double, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::svd, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::svd, task::dim_reduction>;
} // namespace oneapi::dal::pca::parameters
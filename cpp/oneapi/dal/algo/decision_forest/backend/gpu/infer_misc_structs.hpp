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

#pragma once

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::decision_forest::backend {

template <typename Index, typename Task>
struct infer_impl_const;

template <typename Index>
struct infer_impl_const<Index, task::classification> {
    constexpr static Index bad_val = -1;
    constexpr static Index leaf_mark = bad_val;
};

template <typename Index>
struct infer_impl_const<Index, task::regression> {
    constexpr static Index bad_val = -1;
    constexpr static Index leaf_mark = bad_val;
};

template <typename Float, typename Index = std::int32_t, typename Task = task::by_default>
struct infer_context {
    Index class_count = 0;
    Index row_count = 0;
    Index column_count = 0;
    Index tree_count = 0;
    Index tree_in_group_count = 0;
    Index row_block_count = 0;
    voting_mode voting_mode;

    static constexpr inline Index max_local_size = 128;
    static constexpr inline Index max_group_count = 256;

    // following constants showed best performance on benchmark's datasets
    static constexpr inline Index row_count_large = 500000;
    static constexpr inline Index row_count_medium = 100000;

    static constexpr inline Index row_block_count_for_large = 16;
    static constexpr inline Index row_block_count_for_medium = 8;

    static constexpr inline Index tree_count_large = 192;
    static constexpr inline Index tree_count_medium = 48;
    static constexpr inline Index tree_count_small = 12;

    static constexpr inline Index tree_in_group_count_for_large = 128;
    static constexpr inline Index tree_in_group_count_for_medium = 32;
    static constexpr inline Index tree_in_group_count_for_small = 16;
    static constexpr inline Index tree_in_group_count_min = 8;

    static constexpr inline Index preferable_group_size = 256;
    static constexpr inline Index preferable_sbg_size = 16;
    static constexpr inline Index max_local_block_count = 1024;
};
} // namespace oneapi::dal::decision_forest::backend

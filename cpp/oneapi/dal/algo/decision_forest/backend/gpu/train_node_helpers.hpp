/*******************************************************************************
* Copyright 2022 Intel Corporation
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

enum class destination_type { host, device };

template <typename Index = std::int32_t>
class node {
    static_assert(std::is_integral_v<Index>);

public:
    static constexpr Index get_prop_count() {
        return node_prop_count_;
    }

    static constexpr Index get_medium_node_max_row_count() {
        return medium_node_max_row_count_;
    }
    static constexpr Index get_small_node_max_row_count() {
        return small_node_max_row_count_;
    }
    static constexpr Index get_elementary_node_max_row_count() {
        return elementary_node_max_row_count_;
    }

    static constexpr Index ind_ofs() {
        return ind_ofs_;
    }
    static constexpr Index ind_lrc() {
        return ind_lrc_;
    }
    static constexpr Index ind_fid() {
        return ind_fid_;
    }
    static constexpr Index ind_bin() {
        return ind_bin_;
    }
    static constexpr Index ind_lch_grc() {
        return ind_lch_grc_;
    }
    static constexpr Index ind_win() {
        return ind_win_;
    }
    static constexpr Index ind_grc() {
        return ind_grc_;
    }
    static constexpr Index ind_lch_lrc() {
        return ind_lch_lrc_;
    }

private:
    static constexpr inline Index medium_node_max_row_count_ = 8192;
    static constexpr inline Index small_node_max_row_count_ = 256;
    static constexpr inline Index elementary_node_max_row_count_ = 32;

    // left part rows count, response
    // node_prop_count_ is going to be removed here after migration to node_list_manager
    // node props mapping
    constexpr static Index ind_ofs_ = 0; // property index for local row offset
    constexpr static Index ind_lrc_ = 1; // property index for local row count
    constexpr static Index ind_fid_ = 2; // property index for local row count
    constexpr static Index ind_bin_ = 3; // property index for local row count
    constexpr static Index ind_lch_grc_ = 4; // property index for left child global row count
    constexpr static Index ind_win_ = 5; // property index for winner class
    constexpr static Index ind_grc_ = 6; // property index for global row count
    constexpr static Index ind_lch_lrc_ = 7; // property index for left child local row count

    static constexpr inline Index node_prop_count_ = 8;
};
} // namespace oneapi::dal::decision_forest::backend

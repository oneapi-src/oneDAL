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

//#define _P(...) do{ \
//    printf(__VA_ARGS__); printf("\n"); fflush(0); \
//    } while(0)
//
//#define _PL(...) do{ \
//    printf(__VA_ARGS__); fflush(0); \
//    } while(0)

namespace oneapi::dal::decision_forest::backend {

template <typename Task, typename Index = std::int32_t>
struct impl_const;

template <typename Index>
struct ONEDAL_EXPORT impl_const<task::classification, Index> {
    //constexpr static Index bad_val_ = dal::detail::limits<Index>::max();
    constexpr static Index bad_val_ = -1;
    constexpr static Index leaf_node_ = bad_val_;
    constexpr static Index node_prop_count_ = 6; // rows offset, rows count, ftr id, ftr val(bin),
        // left part rows count, response
    constexpr static Index node_imp_prop_count_ = 1; // impurity
};

} // namespace oneapi::dal::decision_forest::backend

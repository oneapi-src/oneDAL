
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

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename Topology, typename EdgeValue, typename BinsVector>
inline void relax_edges(const Topology& t,
                        const EdgeValue* vals,
                        typename Topology::vertex_type u,
                        EdgeValue delta,
                        EdgeValue* dist,
                        BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        EdgeValue old_dist = dist[v];
        const EdgeValue new_dist = dist[u] + v_w;
        if (new_dist < old_dist) {
            dist[v] = new_dist;
            std::int64_t dest_bin = new_dist / delta;
            if (dest_bin >= local_bins.size()) {
                local_bins.resize(dest_bin + 1);
            }
            local_bins[dest_bin].push_back(v);
        }
    }
}

template <typename EV, typename VT>
struct dist_pred {
    dist_pred(const EV& dist_, const VT& pred_) : dist(dist_), pred(pred_) {}
    EV dist;
    VT pred;
};

template <class T1, class T2>
bool operator==(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <class T1, class T2>
bool operator!=(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return !(lhs == rhs);
}

template <typename Topology, typename EdgeValue, typename DP, typename BinsVector>
inline void relax_edges_with_pred(const Topology& t,
                                  const EdgeValue* vals,
                                  typename Topology::vertex_type u,
                                  EdgeValue delta,
                                  DP* dp,
                                  BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        auto old_dp = dp[v];
        const EdgeValue new_dist = dp[u].dist + v_w;
        if (new_dist < old_dp.dist) {
            dp[v] = dist_pred(new_dist, u);
            std::int64_t dest_bin = new_dist / delta;
            if (dest_bin >= local_bins.size()) {
                local_bins.resize(dest_bin + 1);
            }
            local_bins[dest_bin].push_back(v);
        }
    }
}

template <typename Cpu, typename EdgeValue>
struct delta_stepping {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr);
};

template <typename Cpu, typename EdgeValue>
struct delta_stepping_with_pred {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr);
};

template <typename EdgeValue>
struct delta_stepping_with_pred<dal::backend::cpu_dispatch_sse2, EdgeValue> {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr);
};

template <typename EdgeValue>
struct delta_stepping<dal::backend::cpu_dispatch_sse2, EdgeValue> {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr);
};

template <typename Cpu>
bool nrh_disptcher() {
    return false;
}

template <>
bool nrh_disptcher<dal::backend::cpu_dispatch_sse2>() {
    return true;
}

} // namespace oneapi::dal::preview::shortest_paths::backend

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
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

template <typename Method, typename Task, typename Allocator, typename Topology>
struct traverse_kernel_cpu {
    inline traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                            const detail::descriptor_base<Task>& desc,
                                            const Allocator& alloc,
                                            const Topology& t) const;
};

template <typename Float, typename Task, typename Topology, typename... Param>
struct shortest_paths {
    traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                     const detail::descriptor_base<Task>& desc,
                                     const Topology& t) const;
};

template <typename Allocator, typename Topology>
struct traverse_kernel_cpu<method::delta_stepping, task::one_to_all, Allocator, Topology> {
    inline traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const Allocator& alloc,
        const Topology& t) const {
        return traverse_result<task::one_to_all>();
    }
};

} // namespace oneapi::dal::preview::shortest_paths::detail

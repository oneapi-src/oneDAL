/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/relabel_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Method, typename Task, typename Allocator, typename Topology>
struct vertex_ranking_kernel_cpu {
    inline vertex_ranking_result<Task> operator()(const dal::detail::host_policy& ctx,
                                                  const detail::descriptor_base<Task>& desc,
                                                  const Allocator& alloc,
                                                  const Topology& t) const;
};

struct scalar {};
struct vector {};
struct automatic {};

struct relabeled {};

template <typename Float, typename Task, typename Topology, typename... Param>
struct triangle_counting {
    vertex_ranking_result<Task> operator()(const dal::detail::host_policy& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const Topology& t) const;
};

template <typename Float>
struct triangle_counting<Float,
                         task::local,
                         dal::preview::detail::topology<std::int32_t>,
                         automatic> {
    array<std::int64_t> operator()(const dal::detail::host_policy& ctx,
                                   const dal::preview::detail::topology<std::int32_t>& t,
                                   std::int64_t* triangles_local) const;
};

template <typename Float>
struct triangle_counting<Float,
                         task::global,
                         dal::preview::detail::topology<std::int32_t>,
                         scalar> {
    std::int64_t operator()(const dal::detail::host_policy& ctx,
                            const dal::preview::detail::topology<std::int32_t>& t) const;
};

template <typename Float>
struct triangle_counting<Float,
                         task::global,
                         dal::preview::detail::topology<std::int32_t>,
                         vector> {
    std::int64_t operator()(const dal::detail::host_policy& ctx,
                            const dal::preview::detail::topology<std::int32_t>& t) const;
};

template <typename Float>
struct triangle_counting<Float,
                         task::global,
                         dal::preview::detail::topology<std::int32_t>,
                         vector,
                         relabeled> {
    std::int64_t operator()(const dal::detail::host_policy& ctx,
                            const std::int32_t* vertex_neighbors,
                            const std::int64_t* edge_offsets,
                            const std::int32_t* degrees,
                            std::int64_t vertex_count,
                            std::int64_t edge_count) const;
};

ONEDAL_EXPORT std::int64_t compute_global_triangles(const dal::detail::host_policy& policy,
                                                    const array<std::int64_t>& local_triangles,
                                                    std::int64_t vertex_count);

template <typename Allocator, typename Topology>
struct triangle_counting_local {
    inline array<std::int64_t> operator()(const dal::detail::host_policy& ctx,
                                          const Allocator& alloc,
                                          const Topology& t) {
        const auto vertex_count = t.get_vertex_count();

        std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

        using int64_allocator_type =
            typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;

        int64_allocator_type int64_allocator(alloc);

        std::int64_t* triangles_local = oneapi::dal::preview::detail::allocate(
            int64_allocator,
            (std::int64_t)thread_cnt * (std::int64_t)vertex_count);

        auto arr_triangles =
            triangle_counting<float, task::local, Topology, automatic>{}(ctx, t, triangles_local);

        oneapi::dal::preview::detail::deallocate(
            int64_allocator,
            triangles_local,
            (std::int64_t)thread_cnt * (std::int64_t)vertex_count);

        return arr_triangles;
    }
};

template <typename Allocator, typename Topology>
struct vertex_ranking_kernel_cpu<method::ordered_count, task::global, Allocator, Topology> {
    inline vertex_ranking_result<task::global> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::global>& desc,
        const Allocator& alloc,
        const Topology& t) const {
        const auto vertex_count = t.get_vertex_count();
        if (vertex_count == 0) {
            return vertex_ranking_result<task::global>();
        }
        const auto edge_count = t.get_edge_count();
        const auto relabel = desc.get_relabel();
        std::int64_t triangles = 0;
        if (relabel == relabel::yes) {
            const std::int32_t average_degree = edge_count / vertex_count;
            const std::int32_t average_degree_sparsity_boundary = 4;
            if (average_degree < average_degree_sparsity_boundary) {
                triangles = triangle_counting<float, task::global, Topology, scalar>()(ctx, t);
            }
            else {
                std::int32_t* g_vertex_neighbors_relabel = nullptr;
                std::int64_t* g_edge_offsets_relabel = nullptr;
                std::int32_t* g_degrees_relabel = nullptr;

                using int32_allocator_type =
                    typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;

                using int64_allocator_type =
                    typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;

                int64_allocator_type int64_allocator(alloc);
                int32_allocator_type int32_allocator(alloc);

                g_vertex_neighbors_relabel =
                    oneapi::dal::preview::detail::allocate(int32_allocator, edge_count * 2);
                g_degrees_relabel =
                    oneapi::dal::preview::detail::allocate(int32_allocator, vertex_count);
                g_edge_offsets_relabel =
                    oneapi::dal::preview::detail::allocate(int64_allocator, vertex_count + 1);

                relabel_by_greater_degree<Allocator>{}(ctx,
                                                       t,
                                                       g_vertex_neighbors_relabel,
                                                       g_edge_offsets_relabel,
                                                       g_degrees_relabel,
                                                       alloc);

                triangles = triangle_counting<float, task::global, Topology, vector, relabeled>()(
                    ctx,
                    g_vertex_neighbors_relabel,
                    g_edge_offsets_relabel,
                    g_degrees_relabel,
                    vertex_count,
                    edge_count);

                oneapi::dal::preview::detail::deallocate(int32_allocator,
                                                         g_vertex_neighbors_relabel,
                                                         edge_count * 2);
                oneapi::dal::preview::detail::deallocate(int32_allocator,
                                                         g_degrees_relabel,
                                                         vertex_count);

                oneapi::dal::preview::detail::deallocate(int64_allocator,
                                                         g_edge_offsets_relabel,
                                                         vertex_count + 1);
            }
        }
        else {
            const std::int32_t average_degree = edge_count / vertex_count;
            const std::int32_t average_degree_sparsity_boundary = 4;
            if (average_degree < average_degree_sparsity_boundary) {
                triangles = triangle_counting<float, task::global, Topology, scalar>()(ctx, t);
            }
            else {
                triangles = triangle_counting<float, task::global, Topology, vector>()(ctx, t);
            }
        }

        vertex_ranking_result<task::global> res;
        res.set_global_rank(triangles);
        return res;
    }
};

template <typename Allocator, typename Topology>
struct vertex_ranking_kernel_cpu<method::ordered_count, task::local, Allocator, Topology> {
    inline vertex_ranking_result<task::local> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::local>& desc,
        const Allocator& alloc,
        const Topology& t) const {
        if (t.get_vertex_count() == 0) {
            return vertex_ranking_result<task::local>();
        }
        auto local_triangles = triangle_counting_local<Allocator, Topology>()(ctx, alloc, t);

        return vertex_ranking_result<task::local>().set_ranks(
            dal::detail::homogen_table_builder{}
                .reset(local_triangles, t.get_vertex_count(), 1)
                .build());
    }
};

template <typename Allocator, typename Topology>
struct vertex_ranking_kernel_cpu<method::ordered_count,
                                 task::local_and_global,
                                 Allocator,
                                 Topology> {
    inline vertex_ranking_result<task::local_and_global> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::local_and_global>& desc,
        const Allocator& alloc,
        const Topology& t) const {
        const auto vertex_count = t.get_vertex_count();
        if (vertex_count == 0) {
            return vertex_ranking_result<task::local_and_global>();
        }
        auto local_triangles = triangle_counting_local<Allocator, Topology>()(ctx, alloc, t);

        std::int64_t total_s = compute_global_triangles(ctx, local_triangles, vertex_count);

        return vertex_ranking_result<task::local_and_global>()
            .set_ranks(dal::detail::homogen_table_builder{}
                           .reset(local_triangles, vertex_count, 1)
                           .build())
            .set_global_rank(total_s);
    }
};

} // namespace oneapi::dal::preview::triangle_counting::detail

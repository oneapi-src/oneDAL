/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Index>
ONEDAL_EXPORT std::int64_t triangle_counting_global_scalar(const dal::detail::host_policy& policy,
                                                           const Index* vertex_neighbors,
                                                           const std::int64_t* edge_offsets,
                                                           const Index* degrees,
                                                           std::int64_t vertex_count,
                                                           std::int64_t edge_count);

template <typename Index>
ONEDAL_EXPORT std::int64_t triangle_counting_global_vector(const dal::detail::host_policy& policy,
                                                           const Index* vertex_neighbors,
                                                           const std::int64_t* edge_offsets,
                                                           const Index* degrees,
                                                           std::int64_t vertex_count,
                                                           std::int64_t edge_count);

template <typename Index>
ONEDAL_EXPORT std::int64_t triangle_counting_global_vector_relabel(
    const dal::detail::host_policy& policy,
    const Index* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const Index* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count);

template <typename Index>
ONEDAL_EXPORT array<std::int64_t> triangle_counting_local(
    const dal::detail::host_policy& policy,
    const dal::preview::detail::topology<Index>& data,
    int64_t* triangles_local);

ONEDAL_EXPORT std::int64_t compute_global_triangles(const dal::detail::host_policy& policy,
                                                    const array<std::int64_t>& local_triangles,
                                                    std::int64_t vertex_count);

ONEDAL_EXPORT void sort_ids_by_degree(const dal::detail::host_policy& policy,
                                      const std::int32_t* degrees,
                                      std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                                      std::int64_t vertex_count);

ONEDAL_EXPORT void fill_new_degrees_and_ids(
    const dal::detail::host_policy& policy,
    const std::pair<std::int32_t, std::size_t>* degree_id_pairs,
    std::int32_t* new_ids,
    std::int32_t* degrees_relabel,
    std::int64_t vertex_count);

ONEDAL_EXPORT void parallel_prefix_sum(const dal::detail::host_policy& policy,
                                       const std::int32_t* degrees_relabel,
                                       std::int64_t* offsets,
                                       std::int64_t* part_prefix,
                                       std::int64_t* local_sums,
                                       std::int64_t block_size,
                                       std::int64_t num_blocks,
                                       std::int64_t vertex_count);

ONEDAL_EXPORT void fill_relabeled_topology(const dal::detail::host_policy& policy,
                                           const std::int32_t* vertex_neighbors,
                                           const std::int64_t* edge_offsets,
                                           std::int32_t* vertex_neighbors_relabel,
                                           std::int64_t* edge_offsets_relabel,
                                           std::int64_t* offsets,
                                           const std::int32_t* new_ids,
                                           std::int64_t vertex_count);

template <typename Allocator>
inline void relabel_by_greater_degree(const dal::detail::host_policy& ctx,
                                      const std::int32_t* vertex_neighbors,
                                      const std::int64_t* edge_offsets,
                                      const std::int32_t* degrees,
                                      std::int64_t vertex_count,
                                      std::int64_t edge_count,
                                      std::int32_t* vertex_neighbors_relabel,
                                      std::int64_t* edge_offsets_relabel,
                                      std::int32_t* degrees_relabel,
                                      const Allocator& alloc) {
    using int32_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;

    using pair_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
        std::pair<std::int32_t, std::size_t>>;

    using int64_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;

    int64_allocator_type int64_allocator(alloc);
    pair_allocator_type pair_allocator(alloc);
    int32_allocator_type int32_allocator(alloc);

    std::pair<std::int32_t, std::size_t>* degree_id_pairs =

        oneapi::dal::preview::detail::allocate(pair_allocator, vertex_count);

    sort_ids_by_degree(ctx, degrees, degree_id_pairs, vertex_count);

    std::int32_t* new_ids = oneapi::dal::preview::detail::allocate(int32_allocator, vertex_count);

    fill_new_degrees_and_ids(ctx, degree_id_pairs, new_ids, degrees_relabel, vertex_count);

    oneapi::dal::preview::detail::deallocate(pair_allocator, degree_id_pairs, vertex_count);

    std::int64_t* offsets =
        oneapi::dal::preview::detail::allocate(int64_allocator, vertex_count + 1);

    const size_t block_size = 1 << 20;
    const std::int64_t num_blocks = (vertex_count + block_size - 1) / block_size;
    std::int64_t* local_sums = oneapi::dal::preview::detail::allocate(int64_allocator, num_blocks);
    std::int64_t* part_prefix =
        oneapi::dal::preview::detail::allocate(int64_allocator, num_blocks + 1);

    parallel_prefix_sum(ctx,
                        degrees_relabel,
                        offsets,
                        part_prefix,
                        local_sums,
                        block_size,
                        num_blocks,
                        vertex_count);

    oneapi::dal::preview::detail::deallocate(int64_allocator, local_sums, num_blocks);
    oneapi::dal::preview::detail::deallocate(int64_allocator, part_prefix, num_blocks + 1);

    fill_relabeled_topology(ctx,
                            vertex_neighbors,
                            edge_offsets,
                            vertex_neighbors_relabel,
                            edge_offsets_relabel,
                            offsets,
                            new_ids,
                            vertex_count);

    oneapi::dal::preview::detail::deallocate(int64_allocator, offsets, vertex_count + 1);
    oneapi::dal::preview::detail::deallocate(int32_allocator, new_ids, vertex_count);
}

template <typename Allocator>
inline vertex_ranking_result<task::global> triangle_counting_default_kernel(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::global>& desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t>& data) {
    const auto g_edge_offsets = data._rows.get_data();
    const auto g_vertex_neighbors = data._cols.get_data();
    const auto g_degrees = data._degrees.get_data();
    const auto g_vertex_count = data._vertex_count;
    const auto g_edge_count = data._edge_count;

    const auto relabel = desc.get_relabel();
    std::int64_t triangles = 0;

    if (relabel == relabel::yes) {
        std::int32_t average_degree = g_edge_count / g_vertex_count;
        const std::int32_t average_degree_sparsity_boundary = 4;
        if (average_degree < average_degree_sparsity_boundary) {
            triangles = triangle_counting_global_scalar(ctx,
                                                        g_vertex_neighbors,
                                                        g_edge_offsets,
                                                        g_degrees,
                                                        g_vertex_count,
                                                        g_edge_count);
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
                oneapi::dal::preview::detail::allocate(int32_allocator,
                                                       g_edge_offsets[g_vertex_count]);
            g_degrees_relabel =
                oneapi::dal::preview::detail::allocate(int32_allocator, g_vertex_count);
            g_edge_offsets_relabel =
                oneapi::dal::preview::detail::allocate(int64_allocator, g_vertex_count + 1);

            relabel_by_greater_degree(ctx,
                                      g_vertex_neighbors,
                                      g_edge_offsets,
                                      g_degrees,
                                      g_vertex_count,
                                      g_edge_count,
                                      g_vertex_neighbors_relabel,
                                      g_edge_offsets_relabel,
                                      g_degrees_relabel,
                                      alloc);

            triangles = triangle_counting_global_vector_relabel(ctx,
                                                                g_vertex_neighbors_relabel,
                                                                g_edge_offsets_relabel,
                                                                g_degrees_relabel,
                                                                g_vertex_count,
                                                                g_edge_count);

            oneapi::dal::preview::detail::deallocate(int32_allocator,
                                                     g_vertex_neighbors_relabel,
                                                     g_edge_count);
            oneapi::dal::preview::detail::deallocate(int32_allocator,
                                                     g_degrees_relabel,
                                                     g_vertex_count);

            oneapi::dal::preview::detail::deallocate(int64_allocator,
                                                     g_edge_offsets_relabel,
                                                     g_vertex_count + 1);
        }
    }
    else {
        std::int32_t average_degree = g_edge_count / g_vertex_count;
        const std::int32_t average_degree_sparsity_boundary = 4;
        if (average_degree < average_degree_sparsity_boundary) {
            triangles = triangle_counting_global_scalar(ctx,
                                                        g_vertex_neighbors,
                                                        g_edge_offsets,
                                                        g_degrees,
                                                        g_vertex_count,
                                                        g_edge_count);
        }
        else {
            triangles = triangle_counting_global_vector(ctx,
                                                        g_vertex_neighbors,
                                                        g_edge_offsets,
                                                        g_degrees,
                                                        g_vertex_count,
                                                        g_edge_count);
        }
    }

    vertex_ranking_result<task::global> res;
    res.set_global_rank(triangles);
    return res;
}

template <typename Allocator>
inline array<std::int64_t> triangle_counting_local_default_kernel(
    const dal::detail::host_policy& ctx,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t>& data) {
    const auto g_vertex_count = data._vertex_count;

    int thread_cnt = dal::detail::threader_get_max_threads();

    using int64_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;

    int64_allocator_type int64_allocator(alloc);

    int64_t* triangles_local =
        oneapi::dal::preview::detail::allocate(int64_allocator,
                                               (int64_t)thread_cnt * (int64_t)g_vertex_count);

    auto arr_triangles = triangle_counting_local(ctx, data, triangles_local);

    oneapi::dal::preview::detail::deallocate(int64_allocator,
                                             triangles_local,
                                             (int64_t)thread_cnt * (int64_t)g_vertex_count);

    return arr_triangles;
}

template <typename Allocator>
inline vertex_ranking_result<task::local> triangle_counting_default_kernel(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::local>& desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t>& data) {
    auto local_triangles = triangle_counting_local_default_kernel(ctx, alloc, data);

    return vertex_ranking_result<task::local>().set_ranks(
        dal::detail::homogen_table_builder{}.reset(local_triangles, data._vertex_count, 1).build());
}

template <typename Allocator>
inline vertex_ranking_result<task::local_and_global> triangle_counting_default_kernel(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::local_and_global>& desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t>& data) {
    const auto vertex_count = data._vertex_count;

    auto local_triangles = triangle_counting_local_default_kernel(ctx, alloc, data);

    std::int64_t total_s = compute_global_triangles(ctx, local_triangles, vertex_count);

    return vertex_ranking_result<task::local_and_global>()
        .set_ranks(
            dal::detail::homogen_table_builder{}.reset(local_triangles, vertex_count, 1).build())
        .set_global_rank(total_s);
}

} // namespace oneapi::dal::preview::triangle_counting::detail

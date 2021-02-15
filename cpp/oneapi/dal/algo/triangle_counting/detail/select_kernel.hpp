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

#include <iostream>

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/detail/threading.hpp"

#include <chrono>

using namespace std::chrono;

namespace oneapi::dal::preview::triangle_counting::detail {

std::int64_t triangle_counting_global_scalar(const dal::detail::host_policy& policy,
                                             const std::int32_t* vertex_neighbors,
                                             const std::int64_t* edge_offsets,
                                             const std::int32_t* degrees,
                                             std::int64_t vertex_count,
                                             std::int64_t edge_count);

std::int64_t triangle_counting_global_vector(const dal::detail::host_policy& policy,
                                             const std::int32_t* vertex_neighbors,
                                             const std::int64_t* edge_offsets,
                                             const std::int32_t* degrees,
                                             std::int64_t vertex_count,
                                             std::int64_t edge_count);

std::int64_t triangle_counting_global_vector_relabel(const dal::detail::host_policy& policy,
                                                     const std::int32_t* vertex_neighbors,
                                                     const std::int64_t* edge_offsets,
                                                     const std::int32_t* degrees,
                                                     std::int64_t vertex_count,
                                                     std::int64_t edge_count);

array<std::int64_t> triangle_counting_local(
    const dal::detail::host_policy& policy,
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local);

void sort_ids_by_degree(const dal::detail::host_policy& policy,
                        const std::int32_t* degrees,
                        std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                        std::int64_t vertex_count);

void fill_new_degrees_and_ids(const dal::detail::host_policy& policy,
                              std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                              std::int32_t* new_ids,
                              std::int32_t* degrees_relabel,
                              std::int64_t vertex_count);

void parallel_prefix_sum(const dal::detail::host_policy& policy,
                         std::int32_t* degrees_relabel,
                         std::int64_t* offsets,
                         std::int64_t* part_prefix,
                         std::int64_t* local_sums,
                         size_t block_size,
                         std::int64_t num_blocks,
                         std::int64_t vertex_count);

void fill_relabeled_topology(const dal::detail::host_policy& policy,
                             const std::int32_t* vertex_neighbors,
                             const std::int64_t* edge_offsets,
                             std::int32_t* vertex_neighbors_relabel,
                             std::int64_t* edge_offsets_relabel,
                             std::int64_t* offsets,
                             std::int32_t* new_ids,
                             std::int64_t vertex_count);

template <typename Allocator>
void relabel_by_greater_degree(const dal::detail::host_policy& ctx,
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
    using int32_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<std::int32_t>;

    using pair_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
        std::pair<std::int32_t, std::size_t>>;
    using pair_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<
        std::pair<std::int32_t, std::size_t>>;

    using int64_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;
    using int64_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<std::int64_t>;

    int64_allocator_type int64_allocator(alloc);
    pair_allocator_type pair_allocator(alloc);
    int32_allocator_type int32_allocator(alloc);

    std::pair<std::int32_t, std::size_t>* degree_id_pairs =

        pair_allocator_traits::allocate(pair_allocator, vertex_count);

    sort_ids_by_degree(ctx, degrees, degree_id_pairs, vertex_count);

    std::int32_t* new_ids = int32_allocator_traits::allocate(int32_allocator, vertex_count);

    fill_new_degrees_and_ids(ctx, degree_id_pairs, new_ids, degrees_relabel, vertex_count);

    pair_allocator_traits::deallocate(pair_allocator, degree_id_pairs, vertex_count);

    std::int64_t* offsets = int64_allocator_traits::allocate(int64_allocator, vertex_count + 1);

    const size_t block_size = 1 << 20;
    const std::int64_t num_blocks = (vertex_count + block_size - 1) / block_size;
    std::int64_t* local_sums = int64_allocator_traits::allocate(int64_allocator, num_blocks);
    std::int64_t* part_prefix = int64_allocator_traits::allocate(int64_allocator, num_blocks + 1);

    parallel_prefix_sum(ctx,
                        degrees_relabel,
                        offsets,
                        part_prefix,
                        local_sums,
                        block_size,
                        num_blocks,
                        vertex_count);

    int64_allocator_traits::deallocate(int64_allocator, local_sums, num_blocks);
    int64_allocator_traits::deallocate(int64_allocator, part_prefix, num_blocks + 1);

    fill_relabeled_topology(ctx,
                            vertex_neighbors,
                            edge_offsets,
                            vertex_neighbors_relabel,
                            edge_offsets_relabel,
                            offsets,
                            new_ids,
                            vertex_count);

    int64_allocator_traits::deallocate(int64_allocator, offsets, vertex_count + 1);
    int32_allocator_traits::deallocate(int32_allocator, new_ids, vertex_count);
}

template <typename Allocator>
vertex_ranking_result<task::global> triangle_counting_default_kernel_int32(
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
            using int32_allocator_traits =
                typename std::allocator_traits<Allocator>::template rebind_traits<std::int32_t>;

            using int64_allocator_type =
                typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;
            using int64_allocator_traits =
                typename std::allocator_traits<Allocator>::template rebind_traits<std::int64_t>;

            int64_allocator_type int64_allocator(alloc);
            int32_allocator_type int32_allocator(alloc);

            g_vertex_neighbors_relabel =
                int32_allocator_traits::allocate(int32_allocator, g_edge_offsets[g_vertex_count]);
            g_degrees_relabel = int32_allocator_traits::allocate(int32_allocator, g_vertex_count);
            g_edge_offsets_relabel =
                int64_allocator_traits::allocate(int64_allocator, g_vertex_count + 1);

            auto start = high_resolution_clock::now();
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

            auto stop = high_resolution_clock::now();
             std::cout <<  " Relabel: "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count()
                  << std::endl;

            start = high_resolution_clock::now();
            triangles = triangle_counting_global_vector_relabel(ctx,
                                                        g_vertex_neighbors_relabel,
                                                        g_edge_offsets_relabel,
                                                        g_degrees_relabel,
                                                        g_vertex_count,
                                                        g_edge_count);
            stop = high_resolution_clock::now();
            std::cout <<  " TC: "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count()
                  << std::endl;

            if (g_vertex_neighbors_relabel != nullptr) {
                int32_allocator_traits::deallocate(int32_allocator,
                                                   g_vertex_neighbors_relabel,
                                                   g_edge_count);
            }

            if (g_degrees_relabel != nullptr) {
                int32_allocator_traits::deallocate(int32_allocator,
                                                   g_degrees_relabel,
                                                   g_vertex_count);
            }

            if (g_edge_offsets_relabel != nullptr) {
                int64_allocator_traits::deallocate(int64_allocator,
                                                   g_edge_offsets_relabel,
                                                   g_vertex_count + 1);
            }
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
vertex_ranking_result<task::local> triangle_counting_default_kernel_int32(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::local>& desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t>& data) {
    const auto g_vertex_count = data._vertex_count;

    int thread_cnt = dal::detail::threader_get_max_threads();

    using int64_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;
    using int64_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<std::int64_t>;

    int64_allocator_type int64_allocator(alloc);

    int64_t* triangles_local =
        int64_allocator_traits::allocate(int64_allocator,
                                         (int64_t)thread_cnt * (int64_t)g_vertex_count);

    auto arr_triangles = triangle_counting_local(ctx, data, triangles_local);

    int64_allocator_traits::deallocate(int64_allocator,
                                       triangles_local,
                                       (int64_t)thread_cnt * (int64_t)g_vertex_count);

    return vertex_ranking_result<task::local>().set_ranks(
        dal::detail::homogen_table_builder{}.reset(arr_triangles, 1, g_vertex_count).build());
}

template <typename Policy, typename Descriptor, typename Topology>
struct backend_base {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(const Policy& ctx,
                                                     const Descriptor& descriptor,
                                                     const Topology& data) = 0;
    virtual ~backend_base() {}
};

template <typename Policy, typename Descriptor, typename Topology>
struct backend_default : public backend_base<Policy, Descriptor, Topology> {
    using task_t = typename Descriptor::task_t;
    virtual vertex_ranking_result<task_t> operator()(const Policy& ctx,
                                                     const Descriptor& descriptor,
                                                     const Topology& data) {
        return call_triangle_counting_default_kernel_general(descriptor, data);
    }
    virtual ~backend_default() {}
};

template <typename Descriptor>
struct backend_default<dal::detail::host_policy,
                       Descriptor,
                       dal::preview::detail::topology<std::int32_t>>
        : public backend_base<dal::detail::host_policy,
                              Descriptor,
                              dal::preview::detail::topology<std::int32_t>> {
    using task_t = typename Descriptor::task_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(
        const dal::detail::host_policy& ctx,
        const Descriptor& descriptor,
        const dal::preview::detail::topology<std::int32_t>& data) {
        return triangle_counting_default_kernel_int32(ctx,
                                                      descriptor,
                                                      descriptor.get_allocator(),
                                                      data);
    }
    virtual ~backend_default() {}
};

template <typename Policy, typename Descriptor, typename Topology>
dal::detail::pimpl<backend_base<Policy, Descriptor, Topology>> get_backend(const Descriptor& desc,
                                                                           const Topology& data) {
    return dal::detail::pimpl<backend_base<Policy, Descriptor, Topology>>(
        new backend_default<Policy, Descriptor, Topology>);
}

} // namespace oneapi::dal::preview::triangle_counting::detail

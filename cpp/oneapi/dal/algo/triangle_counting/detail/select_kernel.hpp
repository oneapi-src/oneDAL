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

namespace oneapi::dal::preview::triangle_counting::detail {

std::int64_t triangle_counting_global_scalar(const dal::detail::host_policy &policy, const std::int32_t* vertex_neighbors, const std::int64_t* edge_offsets, 
                                const std::int32_t* degrees, std::int64_t vertex_count, std::int64_t edge_count);

std::int64_t triangle_counting_global_vector(const dal::detail::host_policy &policy, const std::int32_t* vertex_neighbors, const std::int64_t* edge_offsets, 
                                const std::int32_t* degrees, std::int64_t vertex_count, std::int64_t edge_count);

std::int64_t triangle_counting_global_vector_relabel(const dal::detail::host_policy &policy, const std::int32_t* vertex_neighbors, const std::int64_t* edge_offsets, 
                                const std::int32_t* degrees, std::int64_t vertex_count, std::int64_t edge_count);

template<typename Allocator>
void relabel_by_greater_degree(const std::int32_t* vertex_neighbors, const std::int64_t* edge_offsets, 
                                const std::int32_t* degrees, std::int64_t vertex_count, std::int64_t edge_count,
                                std::int32_t* vertex_neighbors_relabel, std::int64_t* edge_offsets_relabel, 
                                std::int32_t* degrees_relabel, const Allocator& alloc) {
    using int32_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;
    using int32_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::int32_t>;
    
    using pair_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<std::int32_t, std::size_t>>;
    using pair_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::pair<std::int32_t, std::size_t>>;

    using int64_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;
    using int64_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::int64_t>;

    int64_allocator_type int64_allocator(alloc);
    pair_allocator_type pair_allocator(alloc);
    int32_allocator_type int32_allocator(alloc);

    std::pair<std::int32_t, std::size_t>* degree_id_pairs = pair_allocator_traits::allocate(pair_allocator, vertex_count);
//dive to cpp with dispatching or left it in hpp
//dive start
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t n) {
        degree_id_pairs[n] = std::make_pair(degrees[n], (size_t)n);
    });
    dal::detail::parallel_sort(degree_id_pairs, degree_id_pairs + vertex_count);
//dive end

    std::int32_t* degrees_local = int32_allocator_traits::allocate(int32_allocator, vertex_count);
    std::int32_t* new_ids = int32_allocator_traits::allocate(int32_allocator, vertex_count);

//dive to cpp with dispatching or left it in hpp
//dive start
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t n) {
                degrees_local[n] = degree_id_pairs[n].first;
                new_ids[degree_id_pairs[n].second] = n;
    });
//dive end

    pair_allocator_traits::deallocate(pair_allocator, degree_id_pairs, vertex_count);

    std::int64_t* offsets = int64_allocator_traits::allocate(int64_allocator, vertex_count + 1);


    const std::int32_t size_degrees = vertex_count;
    const size_t block_size = 1 << 20; 
    const std::int64_t num_blocks = (size_degrees + block_size - 1) / block_size; 
    std::int64_t* local_sums = int64_allocator_traits::allocate(int64_allocator, num_blocks);

//dive to cpp with dispatching or left it in hpp
//dive start
    dal::detail::threader_for(num_blocks, num_blocks, [&](std::int64_t block) {
        std::int64_t local_sum = 0;
        std::int64_t block_end = std::min((std::int64_t)((block + 1) * block_size), (std::int64_t)size_degrees);
        for (std::int64_t i=block * block_size; i < block_end; i++) {
            local_sum += degrees_local[i];
        }
        local_sums[block] = local_sum;
    });
//dive end
    std::int64_t* part_prefix = int64_allocator_traits::allocate(int64_allocator, num_blocks + 1);

//dive to cpp with dispatching or left it in hpp
//dive start
    std::int64_t total = 0;
    for (size_t block=0; block < num_blocks; block++) {
        part_prefix[block] = total;
        total += local_sums[block];
    }
    part_prefix[num_blocks] = total;
//dive end

    int64_allocator_traits::deallocate(int64_allocator, local_sums, num_blocks);

//dive to cpp with dispatching or left it in hpp
//dive start
    dal::detail::threader_for(num_blocks, num_blocks, [&](std::int64_t block) {
        std::int64_t local_total = part_prefix[block];
        std::int64_t block_end = std::min((std::int64_t)((block + 1) * block_size), (std::int64_t)size_degrees);
        for (std::int64_t i=block * block_size; i < block_end; i++) {
            offsets[i] = local_total;
            local_total += degrees_local[i];
        }
    });
//dive end

    int32_allocator_traits::deallocate(int32_allocator, degrees_local, vertex_count);
    offsets[size_degrees] = part_prefix[num_blocks];     
    int64_allocator_traits::deallocate(int64_allocator, part_prefix, num_blocks + 1);

//dive to cpp with dispatching or left it in hpp
//dive start
    dal::detail::threader_for(vertex_count + 1, vertex_count + 1, [&](std::int32_t n) {
        edge_offsets_relabel[n] =  offsets[n];
    });
     
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t u) {
        for (const std::int32_t* v = vertex_neighbors + edge_offsets[u]; v != vertex_neighbors + edge_offsets[u + 1]; ++v) {
            vertex_neighbors_relabel[offsets[new_ids[u]]++] = new_ids[*v];
        }

        dal::detail::parallel_sort(vertex_neighbors_relabel + edge_offsets_relabel[new_ids[u]], vertex_neighbors_relabel + edge_offsets_relabel[new_ids[u]+1]);
    });

    for (std::int32_t i = 0; i < vertex_count; i++) {
        degrees_relabel[i] = edge_offsets_relabel[i+1] - edge_offsets_relabel[i];
    }
//dive end

    int64_allocator_traits::deallocate(int64_allocator, offsets, vertex_count + 1);
    int32_allocator_traits::deallocate(int32_allocator, new_ids, vertex_count);
}

template <typename Allocator>
vertex_ranking_result<task::global> triangle_counting_default_kernel_int32(
    const dal::detail::host_policy &ctx,
    const detail::descriptor_base<task::global> &desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t> &data) {

    std::cout << "global tc int32" << std::endl;
    const auto g_edge_offsets = data._rows.get_data();
    const auto g_vertex_neighbors = data._cols.get_data();
    const auto g_degrees = data._degrees.get_data();
    const auto g_vertex_count = data._vertex_count;
    const auto g_edge_count = data._edge_count;

    const auto relabel = desc.get_relabel();
    std::int64_t triangles = 0;

    const std::int32_t average_degree_sparsity_boundary = 4;
    if (g_edge_count / g_vertex_count > average_degree_sparsity_boundary /*&& relabel == relabel::yes*/) {
        std::cout << "relabel" << std::endl;
        std::int32_t* g_vertex_neighbors_relabel = nullptr;
        std::int64_t* g_edge_offsets_relabel = nullptr;
        std::int32_t* g_degrees_relabel = nullptr;

        using Allocator = std::allocator<char>;
        using int32_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;
        using int32_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::int32_t>;

        using int64_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;
        using int64_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::int64_t>;

        int64_allocator_type int64_allocator(alloc);
        int32_allocator_type int32_allocator(alloc);

        g_vertex_neighbors_relabel = int32_allocator_traits::allocate(int32_allocator, g_edge_offsets[g_vertex_count]);
        g_degrees_relabel = int32_allocator_traits::allocate(int32_allocator, g_vertex_count);
        g_edge_offsets_relabel = int64_allocator_traits::allocate(int64_allocator, g_vertex_count + 1);

        relabel_by_greater_degree(g_vertex_neighbors, g_edge_offsets, g_degrees, g_vertex_count, g_edge_count,
                                  g_vertex_neighbors_relabel, g_edge_offsets_relabel, g_degrees_relabel, alloc);

        std::int32_t average_degree = g_edge_count / g_vertex_count;
        const std::int32_t average_degree_sparsity_boundary = 4;
        if (average_degree < average_degree_sparsity_boundary) {
            triangles = triangle_counting_global_scalar(ctx, g_vertex_neighbors_relabel, g_edge_offsets_relabel, g_degrees_relabel, g_vertex_count, g_edge_count);
        } else {
            triangles = triangle_counting_global_vector_relabel(ctx, g_vertex_neighbors_relabel, g_edge_offsets_relabel, g_degrees_relabel, g_vertex_count, g_edge_count);
        }

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
    else {
        std::cout << "no relabel" << std::endl;
        std::int32_t average_degree = g_edge_count / g_vertex_count;
        const std::int32_t average_degree_sparsity_boundary = 4;
        if (average_degree < average_degree_sparsity_boundary) {
            triangles = triangle_counting_global_scalar(ctx, g_vertex_neighbors, g_edge_offsets, g_degrees, g_vertex_count, g_edge_count);
        } else {
            triangles = triangle_counting_global_vector(ctx, g_vertex_neighbors, g_edge_offsets, g_degrees, g_vertex_count, g_edge_count);
        }
    }

    vertex_ranking_result<task::global> res;
    res.set_global_rank(triangles);
    return res;
}
template <typename Allocator>
vertex_ranking_result<task::local> triangle_counting_default_kernel_int32(
    const dal::detail::host_policy &ctx,
    const detail::descriptor_base<task::local> &desc,
    const Allocator& alloc,
    const dal::preview::detail::topology<std::int32_t> &data) {
    std::cout << "default kernel int32 local tc with allocator" << std::endl;

    using int32_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;
    using int32_allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<std::int32_t>;

    int32_allocator_type int32_allocator(alloc);

    auto g_vertex_neighbors_relabel = int32_allocator_traits::allocate(int32_allocator, 123);
    int32_allocator_traits::deallocate(int32_allocator, g_vertex_neighbors_relabel, 123);


    vertex_ranking_result<task::local> res;
    return res;
}



template <typename Policy, typename Descriptor, typename Topology>
struct backend_base {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(const Policy &ctx,
                                                const Descriptor &descriptor,
                                                const Topology &data) = 0;
    virtual ~backend_base() {}
};

template <typename Policy, typename Descriptor, typename Topology>
struct backend_default : public backend_base<Policy, Descriptor, Topology> {
    using task_t = typename Descriptor::task_t;
    virtual vertex_ranking_result<task_t> operator()(const Policy &ctx,
                                                const Descriptor &descriptor,
                                                const Topology &data) {
        return call_triangle_counting_default_kernel_general(descriptor, data);
    }
    virtual ~backend_default() {}
};

template <typename Descriptor>
struct backend_default<dal::detail::host_policy, Descriptor,
                       dal::preview::detail::topology<std::int32_t>>
        : public backend_base<dal::detail::host_policy, Descriptor, 
                              dal::preview::detail::topology<std::int32_t>> {
                                using task_t = typename Descriptor::task_t;
                                using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(
        const dal::detail::host_policy &ctx,
        const Descriptor &descriptor,
        const dal::preview::detail::topology<std::int32_t> &data) {
        // const auto& alloc = descriptor.get_allocator();
        allocator_t alloc;
        return triangle_counting_default_kernel_int32(ctx, descriptor, alloc, data);
    }
    virtual ~backend_default() {}
};

template <typename Policy, typename Descriptor, typename Topology>
dal::detail::pimpl<backend_base<Policy, Descriptor, Topology>> get_backend(const Descriptor &desc,
                                                                     const Topology &data) {
    return dal::detail::pimpl<backend_base<Policy, Descriptor, Topology>>(
        new backend_default<Policy, Descriptor, Topology>);
}

} // namespace oneapi::dal::preview::triangle_counting::detail

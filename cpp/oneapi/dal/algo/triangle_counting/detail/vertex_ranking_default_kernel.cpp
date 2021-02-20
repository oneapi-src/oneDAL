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

#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/relabel_kernels.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <>
std::int64_t triangle_counting_global_scalar<std::int32_t>(const dal::detail::host_policy& policy,
                                                           const std::int32_t* vertex_neighbors,
                                                           const std::int64_t* edge_offsets,
                                                           const std::int32_t* degrees,
                                                           std::int64_t vertex_count,
                                                           std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_scalar_<decltype(cpu)>(vertex_neighbors,
                                                                  edge_offsets,
                                                                  degrees,
                                                                  vertex_count,
                                                                  edge_count);
    });
}

template <>
std::int64_t triangle_counting_global_vector<std::int32_t>(const dal::detail::host_policy& policy,
                                                           const std::int32_t* vertex_neighbors,
                                                           const std::int64_t* edge_offsets,
                                                           const std::int32_t* degrees,
                                                           std::int64_t vertex_count,
                                                           std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_vector_<decltype(cpu)>(vertex_neighbors,
                                                                  edge_offsets,
                                                                  degrees,
                                                                  vertex_count,
                                                                  edge_count);
    });
}

template <>
std::int64_t triangle_counting_global_vector_relabel<std::int32_t>(
    const dal::detail::host_policy& policy,
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_vector_relabel_<decltype(cpu)>(vertex_neighbors,
                                                                          edge_offsets,
                                                                          degrees,
                                                                          vertex_count,
                                                                          edge_count);
    });
}

template <>
array<std::int64_t> triangle_counting_local<std::int32_t>(
    const dal::detail::host_policy& policy,
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_local_<decltype(cpu)>(data, triangles_local);
    });
}

void sort_ids_by_degree(const dal::detail::host_policy& policy,
                        const std::int32_t* degrees,
                        std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                        std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return sort_ids_by_degree_<decltype(cpu)>(degrees, degree_id_pairs, vertex_count);
    });
}

void fill_new_degrees_and_ids(const dal::detail::host_policy& policy,
                              std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                              std::int32_t* new_ids,
                              std::int32_t* degrees_relabel,
                              std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return fill_new_degrees_and_ids_<decltype(cpu)>(degree_id_pairs,
                                                           new_ids,
                                                           degrees_relabel,
                                                           vertex_count);
    });
}

void parallel_prefix_sum(const dal::detail::host_policy& policy,
                         std::int32_t* degrees_relabel,
                         std::int64_t* offsets,
                         std::int64_t* part_prefix,
                         std::int64_t* local_sums,
                         size_t block_size,
                         std::int64_t num_blocks,
                         std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return parallel_prefix_sum_<decltype(cpu)>(degrees_relabel,
                                                      offsets,
                                                      part_prefix,
                                                      local_sums,
                                                      block_size,
                                                      num_blocks,
                                                      vertex_count);
    });
}

void fill_relabeled_topology(const dal::detail::host_policy& policy,
                             const std::int32_t* vertex_neighbors,
                             const std::int64_t* edge_offsets,
                             std::int32_t* vertex_neighbors_relabel,
                             std::int64_t* edge_offsets_relabel,
                             std::int64_t* offsets,
                             std::int32_t* new_ids,
                             std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return fill_relabeled_topology_<decltype(cpu)>(vertex_neighbors,
                                                          edge_offsets,
                                                          vertex_neighbors_relabel,
                                                          edge_offsets_relabel,
                                                          offsets,
                                                          new_ids,
                                                          vertex_count);
    });
}

std::int64_t compute_global_triangles(const dal::detail::host_policy& policy,
                                      const array<std::int64_t>& local_triangles,
                                      std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return compute_global_triangles_<decltype(cpu)>(local_triangles, vertex_count);
    });
}

} // namespace oneapi::dal::preview::triangle_counting::detail

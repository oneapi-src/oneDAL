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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Index>
ONEDAL_FORCEINLINE std::int64_t intersection(const Index* neigh_u,
                                             const Index* neigh_v,
                                             Index n_u,
                                             Index n_v) {
    std::int64_t total = 0;
    Index i_u = 0, i_v = 0;
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

template <typename Index>
ONEDAL_FORCEINLINE std::int64_t intersection_local_tc(const Index* neigh_u,
                                                      const Index* neigh_v,
                                                      Index n_u,
                                                      Index n_v,
                                                      std::int64_t* tc,
                                                      std::int64_t tc_size) {
    std::int64_t total = 0;
    Index i_u = 0, i_v = 0;
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v]) {
            total++, tc[neigh_u[i_u]]++;
            i_u++, i_v++;
        }
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

template <typename Cpu>
ONEDAL_FORCEINLINE array<std::int64_t> triangle_counting_local_(
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local) {
    const auto g_edge_offsets = data._rows.get_data();
    const auto g_vertex_neighbors = data._cols.get_data();
    const auto g_degrees = data._degrees.get_data();
    const auto g_vertex_count = data._vertex_count;
    const auto g_edge_count = data._edge_count;
    std::int32_t average_degree = g_edge_count / g_vertex_count;
    int thread_cnt = dal::detail::threader_get_max_threads();

    dal::detail::threader_for(thread_cnt * g_vertex_count,
                              thread_cnt * g_vertex_count,
                              [&](std::int64_t u) {
                                  triangles_local[u] = 0;
                              });

    const std::int32_t average_degree_sparsity_boundary = 4;
    if (average_degree < average_degree_sparsity_boundary) {
        dal::detail::threader_for(g_vertex_count, g_vertex_count, [&](std::int32_t u) {
            for (auto v_ = g_vertex_neighbors + g_edge_offsets[u];
                 v_ != g_vertex_neighbors + g_edge_offsets[u + 1];
                 ++v_) {
                std::int32_t v = *v_;
                if (v > u) {
                    break;
                }
                auto u_neighbors_ptr = g_vertex_neighbors + g_edge_offsets[u];
                for (auto w_ = g_vertex_neighbors + g_edge_offsets[v];
                     v_ != g_vertex_neighbors + g_edge_offsets[v + 1];
                     ++w_) {
                    std::int32_t w = *w_;
                    if (w > v) {
                        break;
                    }
                    while (*u_neighbors_ptr < w) {
                        u_neighbors_ptr++;
                    }
                    if (w == *u_neighbors_ptr) {
                        int thread_id = dal::detail::threader_get_current_thread_index();
                        int64_t indx = (int64_t)thread_id * (int64_t)g_vertex_count;
                        triangles_local[indx + u]++;
                        triangles_local[indx + v]++;
                        triangles_local[indx + w]++;
                    }
                }
            }
        });
    }
    else { //average_degree >= average_degree_sparsity_boundary
        dal::detail::threader_for_simple(g_vertex_count, g_vertex_count, [&](std::int32_t u) {
            if (g_degrees[u] >= 2)
                dal::detail::threader_for_int32ptr(
                    g_vertex_neighbors + g_edge_offsets[u],
                    g_vertex_neighbors + g_edge_offsets[u + 1],
                    [&](const std::int32_t* v_) {
                        std::int32_t v = *v_;
                        if (v <= u) {
                            const std::int32_t* neigh_u = g_vertex_neighbors + g_edge_offsets[u];
                            std::int32_t size_neigh_u =
                                g_vertex_neighbors + g_edge_offsets[u + 1] - neigh_u;
                            const std::int32_t* neigh_v = g_vertex_neighbors + g_edge_offsets[v];
                            ;
                            std::int32_t size_neigh_v =
                                g_vertex_neighbors + g_edge_offsets[v + 1] - neigh_v;
                            std::int32_t new_size_neigh_v;

                            for (new_size_neigh_v = 0; (new_size_neigh_v < size_neigh_v) &&
                                                       (neigh_v[new_size_neigh_v] <= v);
                                 new_size_neigh_v++)
                                ;
                            size_neigh_v = new_size_neigh_v;

                            int thread_id = dal::detail::threader_get_current_thread_index();
                            int64_t indx = (int64_t)thread_id * (int64_t)g_vertex_count;

                            auto tc = intersection_local_tc(neigh_u,
                                                            neigh_v,
                                                            size_neigh_u,
                                                            size_neigh_v,
                                                            triangles_local + indx,
                                                            g_vertex_count);

                            triangles_local[indx + u] += tc;
                            triangles_local[indx + v] += tc;
                        }
                    });
        });
    }

    auto arr_triangles = array<std::int64_t>::empty(g_vertex_count);

    int64_t* triangles_ptr = arr_triangles.get_mutable_data();

    dal::detail::threader_for(g_vertex_count, g_vertex_count, [&](std::int32_t u) {
        triangles_ptr[u] = 0;
    });

    dal::detail::threader_for(g_vertex_count, g_vertex_count, [&](std::int32_t u) {
        for (int j = 0; j < thread_cnt; j++) {
            int64_t idx_glob = (int64_t)j * (int64_t)g_vertex_count;
            triangles_ptr[u] += triangles_local[idx_glob + u];
        }
    });
    return arr_triangles;
}

template <typename Cpu>
ONEDAL_FORCEINLINE std::int64_t triangle_counting_global_scalar_(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        vertex_count,
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                for (auto v_ = vertex_neighbors + edge_offsets[u];
                     v_ != vertex_neighbors + edge_offsets[u + 1];
                     ++v_) {
                    std::int32_t v = *v_;
                    if (v > u) {
                        break;
                    }
                    auto u_neighbors_ptr = vertex_neighbors + edge_offsets[u];
                    for (auto w_ = vertex_neighbors + edge_offsets[v];
                         v_ != vertex_neighbors + edge_offsets[v + 1];
                         ++w_) {
                        std::int32_t w = *w_;
                        if (w > v) {
                            break;
                        }
                        while (*u_neighbors_ptr < w) {
                            u_neighbors_ptr++;
                        }
                        if (w == *u_neighbors_ptr) {
                            tc_u++;
                        }
                    }
                }
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

template <typename Cpu>
ONEDAL_FORCEINLINE std::int64_t triangle_counting_global_vector_(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t_simple(
        vertex_count,
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                if (degrees[u] < 2) {
                    continue;
                }
                const std::int32_t* neigh_u = vertex_neighbors + edge_offsets[u];
                std::int32_t size_neigh_u = vertex_neighbors + edge_offsets[u + 1] - neigh_u;

                tc_u += oneapi::dal::detail::parallel_reduce_int32ptr_int64_t_simple(
                    vertex_neighbors + edge_offsets[u],
                    vertex_neighbors + edge_offsets[u + 1],
                    (std::int64_t)0,
                    [&](const std::int32_t* begin_v,
                        const std::int32_t* end_v,
                        std::int64_t total) -> std::int64_t {
                        for (auto v_ = begin_v; v_ != end_v; ++v_) {
                            std::int32_t v = *v_;

                            if (v > u) {
                                break;
                            }

                            const std::int32_t* neigh_v = vertex_neighbors + edge_offsets[v];
                            std::int32_t size_neigh_v =
                                vertex_neighbors + edge_offsets[v + 1] - neigh_v;

                            std::int32_t new_size_neigh_v = 0;
                            for (new_size_neigh_v = 0; (new_size_neigh_v < size_neigh_v) &&
                                                       (neigh_v[new_size_neigh_v] <= v);
                                 new_size_neigh_v++)
                                ;

                            total += intersection(neigh_u, neigh_v, size_neigh_u, new_size_neigh_v);
                        }
                        return total;
                    },
                    [&](std::int64_t x, std::int64_t y) -> std::int64_t {
                        return x + y;
                    });
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

template <typename Cpu>
ONEDAL_FORCEINLINE std::int64_t triangle_counting_global_vector_relabel_(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t_simple(
        vertex_count,
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                if (degrees[u] < 2) {
                    continue;
                }
                const std::int32_t* neigh_u = vertex_neighbors + edge_offsets[u];
                std::int32_t size_neigh_u = vertex_neighbors + edge_offsets[u + 1] - neigh_u;

                for (auto v_ = vertex_neighbors + edge_offsets[u];
                     v_ != vertex_neighbors + edge_offsets[u + 1];
                     ++v_) {
                    std::int32_t v = *v_;

                    if (v > u) {
                        break;
                    }

                    const std::int32_t* neigh_v = vertex_neighbors + edge_offsets[v];
                    std::int32_t size_neigh_v = vertex_neighbors + edge_offsets[v + 1] - neigh_v;

                    std::int32_t new_size_neigh_v = 0;
                    for (new_size_neigh_v = 0;
                         (new_size_neigh_v < size_neigh_v) && (neigh_v[new_size_neigh_v] <= v);
                         new_size_neigh_v++)
                        ;

                    tc_u += intersection(neigh_u, neigh_v, size_neigh_u, new_size_neigh_v);
                }
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

} // namespace oneapi::dal::preview::triangle_counting::backend

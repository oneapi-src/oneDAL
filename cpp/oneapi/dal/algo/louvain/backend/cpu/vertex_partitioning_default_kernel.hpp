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

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/algo/louvain/backend/cpu/graph.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename vertex_type>
inline void singleton_partition(vertex_type* labels, std::int64_t vertex_count) {
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        labels[v] = v;
    }
}

template <typename vertex_type, typename vertex_allocator_type>
inline std::int64_t reindex_communities(vertex_type* data,
                                        std::int64_t vertex_count,
                                        vertex_allocator_type& vertex_allocator) {
    vertex_type* index = allocate(vertex_allocator, vertex_count);
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        index[v] = -1;
    }
    vertex_type count = 0;
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        if (index[data[v]] == -1) {
            index[data[v]] = count++;
        }
        data[v] = index[data[v]];
    }
    deallocate(vertex_allocator, index, vertex_count);
    return count;
}

template <typename vertex_type, typename EdgeValue>
inline void compress_graph(graph<vertex_type, EdgeValue>& g,
                           std::int64_t community_count,
                           vertex_type* partition) {
    using value_type = EdgeValue;
    using value_allocator_type = inner_alloc<value_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_type* c_neighbors = allocate(g.vertex_allocator, community_count);
    EdgeValue* weights = allocate(g.value_allocator, community_count);
    EdgeValue* c_self_loops = allocate(g.value_allocator, community_count);
    vertex_type* c_rows = allocate(g.vertex_allocator, community_count + 1);
    c_rows[0] = 0;
    for (std::int64_t c = 0; c < community_count; ++c) {
        c_self_loops[c] = 0;
        weights[c] = 0;
    }
    using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
    using ev1v_t = vector_container<EdgeValue, value_allocator_type>;
    using v1a_t = inner_alloc<v1v_t>;
    using v2v_t = vector_container<v1v_t, v1a_t>;

    v1a_t v1a(g.alloc_ptr);
    v2v_t c2v(community_count, v1a);
    v1v_t c_cols(g.vertex_allocator);
    ev1v_t c_vals(g.value_allocator);

    for (int64_t v = 0; v < g.vertex_count; ++v) {
        vertex_type c = partition[v];
        c2v[c].push_back(v);
    }

    std::int64_t neighbor_count = 0;
    for (std::int64_t c = 0; c < community_count; ++c) {
        for (std::int64_t v_index = 0; v_index < c2v[c].size(); ++v_index) {
            std::int32_t v = c2v[c][v_index];
            c_self_loops[c] += g.self_loops[v];
            for (std::int64_t index = g.rows[v]; index < g.rows[v + 1]; ++index) {
                std::int32_t v_to = g.cols[index];
                std::int32_t c_to = partition[v_to];
                EdgeValue v_w = g.vals[index];
                if (c == c_to) {
                    if (v < v_to) {
                        c_self_loops[c] += v_w;
                    }
                }
                else {
                    if (weights[c_to] == 0) {
                        c_neighbors[neighbor_count++] = c_to;
                    }
                    weights[c_to] += v_w;
                }
            }
        }
        c_rows[c + 1] = c_rows[c] + neighbor_count;
        c_cols.resize(c_rows[c + 1]);
        c_vals.resize(c_rows[c + 1]);
        for (std::int64_t index = 0, c_index = c_rows[c]; index < neighbor_count;
             ++index, ++c_index) {
            std::int32_t c_neigh = c_neighbors[index];
            c_cols[c_index] = c_neigh;
            c_vals[c_index] = weights[c_neigh];
            weights[c_neigh] = 0;
        }
        neighbor_count = 0;
    }

    for (std::int64_t c = 0; c < community_count; ++c) {
        g.self_loops[c] = c_self_loops[c];
        g.rows[c + 1] = c_rows[c + 1];
    }
    for (std::int64_t index = 0; index < c_cols.size(); ++index) {
        g.cols[index] = c_cols[index];
        g.vals[index] = c_vals[index];
    }

    deallocate(g.value_allocator, c_self_loops, community_count);
    deallocate(g.value_allocator, weights, community_count);
    deallocate(g.vertex_allocator, c_neighbors, community_count);
    deallocate(g.vertex_allocator, c_rows, community_count + 1);
}

template <typename vertex_type, typename EdgeValue>
inline double init_step(graph<vertex_type, EdgeValue>& g,
                        vertex_type* labels,
                        double resolution,
                        EdgeValue* k,
                        EdgeValue* tot,
                        EdgeValue& m,
                        std::int64_t* community_size) {
    std::int32_t community_count = 0;
    for (std::int64_t v = 0; v < g.vertex_count; ++v) {
        ++community_size[labels[v]];
        community_count = std::max(community_count, labels[v]);
    }
    ++community_count;
    EdgeValue* k_c = allocate(g.value_allocator, community_count);
    EdgeValue* local_self_loops = allocate(g.value_allocator, community_count);
    for (std::int32_t c = 0; c < community_count; ++c) {
        k_c[c] = 0;
        local_self_loops[c] = 0;
    }
    m = 0;
    for (std::int64_t v = 0; v < g.vertex_count; ++v) {
        std::int32_t c = labels[v];
        local_self_loops[c] += g.self_loops[v];
        k_c[c] += g.self_loops[v] * 2;
        k[v] += g.self_loops[v] * 2;
        tot[c] += g.self_loops[v] * 2;
        m += g.self_loops[v];
        for (std::int64_t index = g.rows[v]; index < g.rows[v + 1]; ++index) {
            std::int32_t to = g.cols[index];
            EdgeValue v_w = g.vals[index];
            std::int32_t to_c = labels[to];
            k_c[c] += v_w;
            k[v] += v_w;
            tot[c] += v_w;
            if (v < to) {
                m += v_w;
                if (c == to_c) {
                    local_self_loops[c] += v_w;
                }
            }
        }
    }

    double modularity = 0;
    for (std::int32_t c = 0; c < community_count; ++c) {
        modularity +=
            1.0 / 2 / m * (local_self_loops[c] * 2 - resolution * k_c[c] * k_c[c] / (2.0 * m));
    }
    deallocate(g.value_allocator, k_c, community_count);
    deallocate(g.value_allocator, local_self_loops, community_count);
    return modularity;
}

template <typename vertex_type, typename EdgeValue>
inline double move_nodes(graph<vertex_type, EdgeValue>& g,
                         vertex_type* n2c,
                         bool& changed,
                         double resolution,
                         double accuracy_threshold) {
    using vertex_size_type = std::int64_t;
    ;

    EdgeValue m = 0;
    EdgeValue* k = allocate(g.value_allocator, g.vertex_count);
    EdgeValue* tot = allocate(g.value_allocator, g.vertex_count);
    vertex_size_type* community_size = allocate(g.vertex_size_allocator, g.vertex_count);
    EdgeValue* k_vertex_to = allocate(g.value_allocator, g.vertex_count);
    EdgeValue* neighboring_communities = allocate(g.value_allocator, g.vertex_count);
    for (std::int64_t v = 0; v < g.vertex_count; ++v) {
        k[v] = 0;
        tot[v] = 0;
        community_size[v] = 0;
        k_vertex_to[v] = 0;
        neighboring_communities[v] = 0;
    }

    // calc initial data
    double modularity = init_step(g, n2c, resolution, k, tot, m, community_size);

    // interate over all vertices
    double old_modularity = modularity;
    vertex_type* random_order = allocate(g.vertex_allocator, g.vertex_count);
    for (std::int64_t index = 0; index < g.vertex_count; ++index) {
        random_order[index] = index;
    }
    //std::random_shuffle(random_order.begin(), random_order.end());
    vertex_type* empty_community = allocate(g.vertex_allocator, g.vertex_count);
    std::int64_t empty_count = 0;
    do {
        old_modularity = modularity;
        for (std::int64_t order_index = 0; order_index < g.vertex_count; ++order_index) {
            std::int32_t v = random_order[order_index];
            std::int32_t c_old = n2c[v];

            // calculate sum of weights of edges between vertex and community to move into
            std::int64_t community_count = 0;
            for (std::int64_t index = g.rows[v]; index < g.rows[v + 1]; ++index) {
                std::int32_t to = g.cols[index];
                std::int32_t c = n2c[to];
                EdgeValue v_w = g.vals[index];
                if (k_vertex_to[c] == 0) {
                    neighboring_communities[community_count++] = c;
                }
                k_vertex_to[c] += v_w;
            }

            // remove vertex from the current community
            EdgeValue k_iold = k_vertex_to[c_old];
            tot[c_old] -= k[v];
            double delta_modularity =
                static_cast<double>(k_iold) / m - resolution * tot[c_old] * k[v] / (2.0 * m * m);
            modularity -= delta_modularity;
            std::int32_t move_community = n2c[v];
            --community_size[c_old];
            if (!community_size[c_old]) {
                empty_community[empty_count++] = c_old;
            }
            // optionaly can be removed, but c_old community can be checked twice
            else if (empty_count) {
                neighboring_communities[community_count++] = empty_community[empty_count - 1];
            }

            // iterate over nodes
            for (std::int32_t index = 0; index < community_count; ++index) {
                std::int32_t c = neighboring_communities[index];

                // try to move vertex to the community
                EdgeValue k_ic = k_vertex_to[c];
                double delta =
                    static_cast<double>(k_ic) / m - resolution * tot[c] * k[v] / (2.0 * m * m);
                if (delta_modularity < delta) {
                    delta_modularity = delta;
                    move_community = c;
                }
                k_vertex_to[c] = 0;
            }
            k_vertex_to[c_old] = 0;

            // move vertex to the best community with the best modularity gain
            modularity += delta_modularity;
            tot[move_community] += k[v];
            n2c[v] = move_community;
            if (move_community != c_old) {
                changed = true;
            }
            ++community_size[move_community];
            if (community_size[move_community] == 1) {
                --empty_count;
            }
        }
    } while (modularity - old_modularity > accuracy_threshold);

    deallocate(g.value_allocator, k, g.vertex_count);
    deallocate(g.value_allocator, tot, g.vertex_count);
    deallocate(g.vertex_size_allocator, community_size, g.vertex_count);
    deallocate(g.value_allocator, k_vertex_to, g.vertex_count);
    deallocate(g.value_allocator, neighboring_communities, g.vertex_count);
    deallocate(g.vertex_allocator, random_order, g.vertex_count);
    deallocate(g.vertex_allocator, empty_community, g.vertex_count);

    return modularity;
}

template <typename vertex_type, typename community_vector_type, typename size_vector_type>
inline void set_result_labels(community_vector_type& communities,
                              size_vector_type& vertex_size,
                              const vertex_type* init_partition,
                              std::int64_t vertex_count,
                              vertex_type* result_labels) {
    if (!communities.empty()) {
        // flat the communities from the next iteration
        for (std::int64_t iteration = communities.size() - 2; iteration >= 0; --iteration) {
            for (std::int64_t v = 0; v < vertex_size[iteration]; ++v) {
                communities[iteration][v] = communities[iteration + 1][communities[iteration][v]];
            }
        }
        for (std::int64_t v = 0; v < vertex_count; ++v) {
            result_labels[v] = communities[0][v];
        }
    }
    else if (init_partition == nullptr) {
        for (std::int64_t v = 0; v < vertex_count; ++v) {
            result_labels[v] = v;
        }
    }
    else {
        for (std::int64_t v = 0; v < vertex_count; ++v) {
            result_labels[v] = init_partition[v];
        }
    }
}

template <typename Cpu, typename EdgeValue>
struct louvain_kernel {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const std::int32_t* init_partition,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        {
            using value_type = EdgeValue;
            using vertex_type = std::int32_t;
            using vertex_size_type = std::int64_t;
            using value_allocator_type = inner_alloc<value_type>;
            using vertex_allocator_type = inner_alloc<vertex_type>;
            using vertex_size_allocator_type = inner_alloc<vertex_size_type>;

            double resolution = desc.get_resolution();
            double accuracy_threshold = desc.get_accuracy_threshold();
            std::int64_t max_iteration_count = desc.get_max_iteration_count();

            auto vertex_count = t.get_vertex_count();
            graph<std::int32_t, EdgeValue> g(t, vals, alloc_ptr);

            double modularity = std::numeric_limits<double>::min();
            vertex_type* labels = allocate(g.vertex_allocator, vertex_count);
            if (init_partition != nullptr) {
                for (std::int64_t v = 0; v < vertex_count; ++v) {
                    labels[v] = init_partition[v];
                }
            }
            else {
                singleton_partition(labels, vertex_count);
            }

            using vertex_p_type = vertex_type*;
            using vertex_p_allocator_type = inner_alloc<vertex_p_type>;
            using vp_t = vector_container<vertex_p_type, vertex_p_allocator_type>;
            using vs_t = vector_container<vertex_size_type, vertex_size_allocator_type>;

            vertex_p_allocator_type vp_a(alloc_ptr);
            vp_t communities(vp_a);
            vs_t labels_size(g.vertex_size_allocator);
            vs_t vertex_size(g.vertex_size_allocator);

            bool allocate_labels = false;
            for (std::int64_t iteration = 0;
                 iteration < max_iteration_count || !max_iteration_count;
                 ++iteration) {
                if (allocate_labels) {
                    labels = allocate(g.vertex_allocator, vertex_count);
                    singleton_partition(labels, vertex_count);
                }
                allocate_labels = true;
                bool changed = false;
                modularity = move_nodes(g, labels, changed, resolution, accuracy_threshold);

                if (!changed) {
                    deallocate(g.vertex_allocator, labels, vertex_count);
                    break;
                }
                std::int64_t community_count =
                    reindex_communities(labels, g.vertex_count, g.vertex_allocator);

                compress_graph(g, community_count, labels);
                labels_size.push_back(community_count);
                vertex_size.push_back(g.vertex_count);
                communities.push_back(labels);
                g.vertex_count = community_count;
            }

            auto labels_arr = array<vertex_type>::empty(t.get_vertex_count());
            vertex_type* labels_ = labels_arr.get_mutable_data();
            set_result_labels(communities,
                              vertex_size,
                              init_partition,
                              t.get_vertex_count(),
                              labels_);

            for (int64_t iteration = 0; iteration < communities.size(); ++iteration) {
                deallocate(g.vertex_allocator, communities[iteration], labels_size[iteration]);
            }

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(labels_arr, t.get_vertex_count(), 1)
                                .build())
                .set_modularity(modularity)
                .set_community_count(g.vertex_count);
        }
    }
};

} // namespace oneapi::dal::preview::louvain::backend

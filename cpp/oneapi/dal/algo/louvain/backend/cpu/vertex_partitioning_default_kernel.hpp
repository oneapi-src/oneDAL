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
#include "oneapi/dal/backend/primitives/rng/random_shuffle.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;
using namespace oneapi::dal::backend::primitives;

template <typename IndexType>
inline void singleton_partition(IndexType* labels, const std::int64_t vertex_count) {
    for (std::int64_t v = 0; v < vertex_count; v++) {
        labels[v] = v;
    }
}

template <typename IndexType>
inline std::int64_t reindex_communities(IndexType* communities,
                                        const std::int64_t vertex_count,
                                        IndexType* index) {
    for (std::int64_t v = 0; v < vertex_count; v++) {
        index[v] = -1;
    }
    IndexType count = 0;
    for (std::int64_t v = 0; v < vertex_count; v++) {
        if (index[communities[v]] == -1) {
            index[communities[v]] = count++;
        }
        communities[v] = index[communities[v]];
    }
    return count;
}

template <typename IndexType, typename EdgeValue, typename CommunityVertexContainer>
inline void compress_graph(dal::preview::detail::topology<std::int32_t>& t,
                           EdgeValue* vals,
                           EdgeValue* self_loops,
                           const std::int64_t& community_count,
                           const IndexType* partition,
                           const std::int64_t* community_size,
                           IndexType* c_neighbors,
                           IndexType* c_rows,
                           EdgeValue* weights,
                           EdgeValue* c_self_loops,
                           CommunityVertexContainer& c2v,
                           IndexType* c_cols,
                           EdgeValue* c_vals) {
    c_rows[0] = 0;
    for (std::int64_t c = 0; c < community_count; c++) {
        c2v[c].resize(0);
        c2v[c].reserve(community_size[c]);
        c_self_loops[c] = 0;
        weights[c] = 0;
    }
    for (int64_t v = 0; v < t._vertex_count; v++) {
        std::int32_t c = partition[v];
        c2v[c].push_back(v);
    }

    std::int64_t neighbor_count = 0;
    for (std::int64_t c = 0; c < community_count; c++) {
        for (std::int64_t v_index = 0; v_index < c2v[c].size(); v_index++) {
            std::int32_t v = c2v[c][v_index];
            c_self_loops[c] += self_loops[v];
            for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
                std::int32_t v_to = t._cols_ptr[index];
                std::int32_t c_to = partition[v_to];
                EdgeValue v_w = vals[index];
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
        for (std::int64_t index = 0, c_index = c_rows[c]; index < neighbor_count;
             index++, c_index++) {
            std::int32_t c_neigh = c_neighbors[index];
            c_cols[c_index] = c_neigh;
            c_vals[c_index] = weights[c_neigh];
            weights[c_neigh] = 0;
        }
        neighbor_count = 0;
    }

    std::int64_t* t_rows = t._rows.get_mutable_data();
    for (std::int64_t c = 0; c < community_count; c++) {
        self_loops[c] = c_self_loops[c];
        t_rows[c + 1] = c_rows[c + 1];
    }
    IndexType* t_cols = t._cols.get_mutable_data();
    for (std::int64_t index = 0; index < c_rows[community_count]; index++) {
        t_cols[index] = c_cols[index];
        vals[index] = c_vals[index];
    }
}

template <typename Float, typename IndexType, typename EdgeValue>
inline Float init_step(const dal::preview::detail::topology<std::int32_t>& t,
                       const EdgeValue* vals,
                       const EdgeValue* self_loops,
                       const IndexType* labels,
                       const double resolution,
                       EdgeValue* k,
                       EdgeValue* tot,
                       EdgeValue& m,
                       std::int64_t* community_size,
                       EdgeValue* k_c,
                       EdgeValue* local_self_loops) {
    std::int32_t community_count = 0;
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        community_size[labels[v]]++;
        community_count = std::max(community_count, labels[v]);
    }
    community_count++;
    for (std::int32_t c = 0; c < community_count; c++) {
        k_c[c] = 0;
        local_self_loops[c] = 0;
    }
    m = 0;
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        std::int32_t c = labels[v];
        local_self_loops[c] += self_loops[v];
        k_c[c] += self_loops[v] * 2;
        k[v] += self_loops[v] * 2;
        tot[c] += self_loops[v] * 2;
        m += self_loops[v];
        for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
            std::int32_t to = t._cols_ptr[index];
            EdgeValue v_w = vals[index];
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
    ONEDAL_ASSERT(m > 0);

    Float modularity = 0;
    for (std::int32_t c = 0; c < community_count; c++) {
        modularity +=
            1.0 / 2 / m * (local_self_loops[c] * 2 - resolution * k_c[c] * k_c[c] / (2.0 * m));
    }
    return modularity;
}

template <typename Cpu, typename Float, typename IndexType, typename EdgeValue>
inline Float move_nodes(const dal::preview::detail::topology<std::int32_t>& t,
                        const EdgeValue* vals,
                        const EdgeValue* self_loops,
                        IndexType* n2c,
                        bool& changed,
                        const double resolution,
                        const double accuracy_threshold,
                        EdgeValue* k,
                        EdgeValue* tot,
                        EdgeValue* k_vertex_to,
                        IndexType* neighboring_communities,
                        IndexType* random_order,
                        IndexType* empty_community,
                        std::int64_t* community_size,
                        EdgeValue* k_c,
                        EdgeValue* local_self_loops) {
    EdgeValue m = 0;
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        k[v] = 0;
        tot[v] = 0;
        community_size[v] = 0;
        k_vertex_to[v] = 0;
        neighboring_communities[v] = 0;
    }

    // calc initial data
    Float modularity = init_step<Float>(t,
                                        vals,
                                        self_loops,
                                        n2c,
                                        resolution,
                                        k,
                                        tot,
                                        m,
                                        community_size,
                                        k_c,
                                        local_self_loops);

    // interate over all vertices
    Float old_modularity = modularity;
    for (std::int64_t index = 0; index < t._vertex_count; index++) {
        random_order[index] = index;
    }
    random_shuffle<Cpu>(random_order, t._vertex_count);
    std::int64_t empty_count = 0;
    do {
        old_modularity = modularity;
        for (std::int64_t order_index = 0; order_index < t._vertex_count; order_index++) {
            std::int32_t v = random_order[order_index];
            std::int32_t c_old = n2c[v];

            // calculate sum of weights of edges between vertex and community to move into
            std::int64_t community_count = 0;
            for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
                std::int32_t to = t._cols_ptr[index];
                std::int32_t c = n2c[to];
                EdgeValue v_w = vals[index];
                if (k_vertex_to[c] == 0) {
                    neighboring_communities[community_count++] = c;
                }
                k_vertex_to[c] += v_w;
            }

            // remove vertex from the current community
            EdgeValue k_iold = k_vertex_to[c_old];
            tot[c_old] -= k[v];
            Float delta_modularity =
                static_cast<Float>(k_iold) / m - resolution * tot[c_old] * k[v] / (2.0 * m * m);
            modularity -= delta_modularity;
            std::int32_t move_community = n2c[v];
            community_size[c_old]--;
            if (!community_size[c_old]) {
                empty_community[empty_count++] = c_old;
            }
            // optionaly can be removed, but c_old community can be checked twice
            else if (empty_count) {
                neighboring_communities[community_count++] = empty_community[empty_count - 1];
            }

            // iterate over nodes
            for (std::int32_t index = 0; index < community_count; index++) {
                std::int32_t c = neighboring_communities[index];

                // try to move vertex to the community
                EdgeValue k_ic = k_vertex_to[c];
                const Float delta =
                    static_cast<Float>(k_ic) / m - resolution * tot[c] * k[v] / (2.0 * m * m);
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
            community_size[move_community]++;
            if (community_size[move_community] == 1) {
                empty_count--;
            }
        }
    } while (modularity - old_modularity > accuracy_threshold);

    return modularity;
}

template <typename IndexType, typename CommunityVector, typename SizeVector>
inline void set_result_labels(CommunityVector& communities,
                              const SizeVector& vertex_size,
                              const IndexType* init_partition,
                              const std::int64_t vertex_count,
                              IndexType* result_labels) {
    if (!communities.empty()) {
        // flat the communities from the next iteration
        for (std::int64_t iteration = communities.size() - 2; iteration >= 0; iteration--) {
            for (std::int64_t v = 0; v < vertex_size[iteration]; v++) {
                communities[iteration][v] = communities[iteration + 1][communities[iteration][v]];
            }
        }
        for (std::int64_t v = 0; v < vertex_count; v++) {
            result_labels[v] = communities[0][v];
        }
    }
    else if (init_partition == nullptr) {
        for (std::int64_t v = 0; v < vertex_count; v++) {
            result_labels[v] = v;
        }
    }
    else {
        for (std::int64_t v = 0; v < vertex_count; v++) {
            result_labels[v] = init_partition[v];
        }
    }
}

template <typename Cpu, typename Float, typename EdgeValue>
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
            using vertex_pointer_type = vertex_type*;

            using value_allocator_type = inner_alloc<value_type>;
            using vertex_allocator_type = inner_alloc<vertex_type>;
            using vertex_size_allocator_type = inner_alloc<vertex_size_type>;
            using vertex_pointer_allocator_type = inner_alloc<vertex_pointer_type>;

            using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
            using v1s_t = vector_container<vertex_size_type, vertex_size_allocator_type>;
            using v1p_t = vector_container<vertex_pointer_type, vertex_pointer_allocator_type>;
            using v1a_t = inner_alloc<v1v_t>;
            using v2v_t = vector_container<v1v_t, v1a_t>;

            vertex_allocator_type vertex_allocator(alloc_ptr);
            vertex_size_allocator_type vertex_size_allocator(alloc_ptr);
            value_allocator_type value_allocator(alloc_ptr);
            vertex_pointer_allocator_type vp_a(alloc_ptr);
            v1a_t v1a(alloc_ptr);

            const double resolution = desc.get_resolution();
            const double accuracy_threshold = desc.get_accuracy_threshold();
            const std::int64_t max_iteration_count = desc.get_max_iteration_count();

            const std::int64_t vertex_count = t.get_vertex_count();
            const std::int64_t edge_count = t.get_edge_count();
            dal::preview::detail::topology<std::int32_t> current_topology;

            auto current_topology_rows_shared_ptr =
                vertex_size_allocator.make_shared_memory(vertex_count + 1);
            auto current_topology_cols_shared_ptr =
                vertex_allocator.make_shared_memory(edge_count * 2);
            auto current_vals_shared_ptr = value_allocator.make_shared_memory(edge_count * 2);
            auto current_self_loops_shared_ptr = value_allocator.make_shared_memory(edge_count * 2);

            vertex_size_type* current_topology_rows = current_topology_rows_shared_ptr.get();
            vertex_type* current_topology_cols = current_topology_cols_shared_ptr.get();
            value_type* current_vals = current_vals_shared_ptr.get();
            value_type* current_self_loops = current_self_loops_shared_ptr.get();

            current_topology.set_topology(vertex_count,
                                          edge_count,
                                          current_topology_rows,
                                          current_topology_cols,
                                          edge_count * 2,
                                          nullptr);

            current_topology_rows[0] = t._rows_ptr[0];
            for (std::int64_t index = 0; index < vertex_count; index++) {
                current_topology_rows[index + 1] = t._rows_ptr[index + 1];
                current_self_loops[index] = 0;
            }
            for (std::int64_t index = 0; index < edge_count * 2; index++) {
                current_topology_cols[index] = t._cols_ptr[index];
                current_vals[index] = vals[index];
            }

            auto k_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto tot_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto k_vertex_to_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto neighboring_communities_shared_ptr =
                vertex_allocator.make_shared_memory(vertex_count);
            auto random_order_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
            auto empty_community_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
            auto community_size_shared_ptr = vertex_size_allocator.make_shared_memory(vertex_count);

            auto k_c_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto local_self_loops_shared_ptr = value_allocator.make_shared_memory(vertex_count);

            auto weights_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto c_self_loops_shared_ptr = value_allocator.make_shared_memory(vertex_count);
            auto c_neighbors_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
            auto c_rows_shared_ptr = vertex_allocator.make_shared_memory(vertex_count + 1);

            auto c_vals_shared_ptr = value_allocator.make_shared_memory(edge_count * 2);
            auto c_cols_shared_ptr = vertex_allocator.make_shared_memory(edge_count * 2);
            auto index_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);

            v2v_t c2v(vertex_count, v1a);
            v1p_t communities(vp_a);
            v1s_t labels_size(vertex_size_allocator);
            v1s_t vertex_size(vertex_size_allocator);

            Float modularity = std::numeric_limits<Float>::min();
            vertex_type* labels = allocate(vertex_allocator, vertex_count);
            if (init_partition != nullptr) {
                for (std::int64_t v = 0; v < vertex_count; v++) {
                    labels[v] = init_partition[v];
                }
            }
            else {
                singleton_partition(labels, vertex_count);
            }

            bool allocate_labels = false;
            for (std::int64_t iteration = 0;
                 iteration < max_iteration_count || !max_iteration_count;
                 iteration++) {
                if (allocate_labels) {
                    labels = allocate(vertex_allocator, current_topology._vertex_count);
                    singleton_partition(labels, current_topology._vertex_count);
                }
                allocate_labels = true;
                bool changed = false;
                modularity = move_nodes<Cpu, Float>(current_topology,
                                                    current_vals,
                                                    current_self_loops,
                                                    labels,
                                                    changed,
                                                    resolution,
                                                    accuracy_threshold,
                                                    k_shared_ptr.get(),
                                                    tot_shared_ptr.get(),
                                                    k_vertex_to_shared_ptr.get(),
                                                    neighboring_communities_shared_ptr.get(),
                                                    random_order_shared_ptr.get(),
                                                    empty_community_shared_ptr.get(),
                                                    community_size_shared_ptr.get(),
                                                    k_c_shared_ptr.get(),
                                                    local_self_loops_shared_ptr.get());

                if (!changed) {
                    deallocate(vertex_allocator, labels, current_topology._vertex_count);
                    break;
                }
                std::int64_t community_count = reindex_communities(labels,
                                                                   current_topology._vertex_count,
                                                                   index_shared_ptr.get());

                compress_graph(current_topology,
                               current_vals,
                               current_self_loops,
                               community_count,
                               labels,
                               community_size_shared_ptr.get(),
                               c_neighbors_shared_ptr.get(),
                               c_rows_shared_ptr.get(),
                               weights_shared_ptr.get(),
                               c_self_loops_shared_ptr.get(),
                               c2v,
                               c_cols_shared_ptr.get(),
                               c_vals_shared_ptr.get());
                labels_size.push_back(community_count);
                vertex_size.push_back(current_topology._vertex_count);
                communities.push_back(labels);
                current_topology._vertex_count = community_count;
            }

            auto labels_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* labels_ = labels_arr.get_mutable_data();
            set_result_labels(communities, vertex_size, init_partition, vertex_count, labels_);

            for (int64_t iteration = 0; iteration < communities.size(); iteration++) {
                deallocate(vertex_allocator, communities[iteration], labels_size[iteration]);
            }

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(labels_arr, t.get_vertex_count(), 1)
                                .build())
                .set_modularity(modularity)
                .set_community_count(current_topology._vertex_count);
        }
    }
};

} // namespace oneapi::dal::preview::louvain::backend

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

#include <unordered_map>

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

template <typename vertex_type,
          typename vertex_size_type,
          typename EdgeValue,
          typename value_allocator_type,
          typename vertex_allocator_type>
inline void compress_graph(vertex_size_type* rows,
                           vertex_type* cols,
                           EdgeValue* vals,
                           EdgeValue* self_loops,
                           std::int64_t vertex_count,
                           std::int64_t community_count,
                           vertex_type* partition,
                           value_allocator_type& value_allocator,
                           vertex_allocator_type& vertex_allocator,
                           byte_alloc_iface* alloc_ptr) {
    std::vector<std::unordered_map<vertex_type, EdgeValue>> weights(community_count);
    EdgeValue* c_self_loops = allocate(value_allocator, community_count);
    for (std::int64_t c = 0; c < community_count; ++c) {
        c_self_loops[c] = 0;
    }
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        vertex_type c = partition[v];
        c_self_loops[c] += self_loops[v];
        for (std::int64_t index = rows[v]; index < rows[v + 1]; ++index) {
            vertex_type to = cols[index];
            EdgeValue v_w = vals[index];
            vertex_type to_c = partition[to];
            if (c == to_c) {
                if (v < to) {
                    c_self_loops[c] += v_w;
                }
            }
            else {
                weights[c][to_c] += v_w;
            }
        }
    }
    std::int64_t cols_size = 0;
    for (std::int64_t index = 0; index < community_count; ++index) {
        rows[index + 1] = rows[index] + weights[index].size();
        cols_size += weights[index].size();
    }
    for (std::int64_t index = 0, c = 0; c < community_count; ++c) {
        for (const auto& edge : weights[c]) {
            cols[index] = edge.first;
            vals[index++] = edge.second;
        }
    }

    for (std::int64_t c = 0; c < community_count; ++c) {
        self_loops[c] = c_self_loops[c];
    }
    deallocate(value_allocator, c_self_loops, community_count);
}

template <typename vertex_type,
          typename vertex_size_type,
          typename EdgeValue,
          typename value_allocator_type>
inline double init_step(vertex_size_type* rows,
                        vertex_type* cols,
                        EdgeValue* vals,
                        EdgeValue* self_loops,
                        std::int64_t vertex_count,
                        vertex_type* labels,
                        double resolution,
                        EdgeValue* k,
                        EdgeValue* tot,
                        EdgeValue& m,
                        vertex_size_type* community_size,
                        value_allocator_type& value_allocator) {
    std::int32_t community_count = 0;
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        ++community_size[labels[v]];
        community_count = std::max(community_count, labels[v]);
    }
    ++community_count;
    EdgeValue* k_c = allocate(value_allocator, community_count);
    EdgeValue* local_self_loops = allocate(value_allocator, community_count);
    for (std::int32_t c = 0; c < community_count; ++c) {
        k_c[c] = 0;
        local_self_loops[c] = 0;
    }
    m = 0;
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        std::int32_t c = labels[v];
        local_self_loops[c] += self_loops[v];
        k_c[c] += self_loops[v] * 2;
        k[v] += self_loops[v] * 2;
        tot[c] += self_loops[v] * 2;
        m += self_loops[v];
        for (std::int64_t index = rows[v]; index < rows[v + 1]; ++index) {
            std::int32_t to = cols[index];
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

    double modularity = 0;
    for (std::int32_t c = 0; c < community_count; ++c) {
        modularity +=
            1.0 / 2 / m * (local_self_loops[c] * 2 - resolution * k_c[c] * k_c[c] / (2.0 * m));
    }
    deallocate(value_allocator, k_c, community_count);
    deallocate(value_allocator, local_self_loops, community_count);
    return modularity;
}

template <typename vertex_type,
          typename vertex_size_type,
          typename EdgeValue,
          typename value_allocator_type,
          typename vertex_size_allocator_type,
          typename vertex_allocator_type>
inline double move_nodes(vertex_size_type* rows,
                         vertex_type* cols,
                         EdgeValue* vals,
                         EdgeValue* self_loops,
                         std::int64_t vertex_count,
                         vertex_type* n2c,
                         bool& changed,
                         double resolution,
                         double accuracy_threshold,
                         value_allocator_type& value_allocator,
                         vertex_size_allocator_type& vertex_size_allocator,
                         vertex_allocator_type& vertex_allocator) {
    EdgeValue m = 0;
    EdgeValue* k = allocate(value_allocator, vertex_count);
    EdgeValue* tot = allocate(value_allocator, vertex_count);
    vertex_size_type* community_size = allocate(vertex_size_allocator, vertex_count);
    EdgeValue* k_vertex_to = allocate(value_allocator, vertex_count);
    EdgeValue* neighboring_communities = allocate(value_allocator, vertex_count);
    for (std::int64_t v = 0; v < vertex_count; ++v) {
        k[v] = 0;
        tot[v] = 0;
        community_size[v] = 0;
        k_vertex_to[v] = 0;
        neighboring_communities[v] = 0;
    }

    // calc initial data
    double modularity = init_step(rows,
                                  cols,
                                  vals,
                                  self_loops,
                                  vertex_count,
                                  n2c,
                                  resolution,
                                  k,
                                  tot,
                                  m,
                                  community_size,
                                  value_allocator);

    // interate over all vertices
    double old_modularity = modularity;
    vertex_type* random_order = allocate(vertex_allocator, vertex_count);
    for (std::int64_t index = 0; index < vertex_count; ++index) {
        random_order[index] = index;
    }
    //std::random_shuffle(random_order.begin(), random_order.end());
    vertex_type* empty_community = allocate(vertex_allocator, vertex_count);
    std::int32_t empty_count = 0;
    do {
        old_modularity = modularity;
        for (std::int32_t order_index = 0; order_index < vertex_count; ++order_index) {
            std::int32_t v = random_order[order_index];
            std::int32_t c_old = n2c[v];

            // calculate sum of weights of edges between vertex and community to move into
            std::int32_t community_count = 0;
            for (std::int64_t index = rows[v]; index < rows[v + 1]; ++index) {
                std::int32_t to = cols[index];
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

    deallocate(value_allocator, k, vertex_count);
    deallocate(value_allocator, tot, vertex_count);
    deallocate(vertex_size_allocator, community_size, vertex_count);
    deallocate(value_allocator, k_vertex_to, vertex_count);
    deallocate(value_allocator, neighboring_communities, vertex_count);
    deallocate(vertex_allocator, random_order, vertex_count);
    deallocate(vertex_allocator, empty_community, vertex_count);

    return modularity;
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

            value_allocator_type value_allocator(alloc_ptr);
            vertex_allocator_type vertex_allocator(alloc_ptr);
            vertex_size_allocator_type vertex_size_allocator(alloc_ptr);

            double resolution = desc.get_resolution();
            double accuracy_threshold = desc.get_accuracy_threshold();
            std::int64_t max_iteration_count = desc.get_max_iteration_count();

            auto vertex_count = t.get_vertex_count();
            auto edge_count = t.get_edge_count();

            vertex_size_type* rows = allocate(vertex_size_allocator, vertex_count + 1);
            vertex_type* cols = allocate(vertex_allocator, edge_count * 2);
            value_type* weights = allocate(value_allocator, edge_count * 2);
            value_type* self_loops = allocate(value_allocator, vertex_count);

            for (std::int64_t index = 0; index <= vertex_count; ++index) {
                rows[index] = t._rows_ptr[index];
            }
            for (std::int64_t index = 0; index < edge_count * 2; ++index) {
                cols[index] = t._cols_ptr[index];
                weights[index] = vals[index];
            }
            for (std::int64_t index = 0; index < vertex_count; ++index) {
                self_loops[index] = 0;
            }

            double modularity = std::numeric_limits<double>::min();
            vertex_type* labels = allocate(vertex_allocator, vertex_count);
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
            vs_t labels_size(vertex_size_allocator);
            vs_t vertex_size(vertex_size_allocator);

            bool allocate_labels = false;
            for (std::int64_t iteration = 0;
                 iteration < max_iteration_count || !max_iteration_count;
                 ++iteration) {
                if (allocate_labels) {
                    labels = allocate(vertex_allocator, vertex_count);
                    singleton_partition(labels, vertex_count);
                }
                allocate_labels = true;
                bool changed = false;
                modularity = move_nodes(rows,
                                        cols,
                                        weights,
                                        self_loops,
                                        vertex_count,
                                        labels,
                                        changed,
                                        resolution,
                                        accuracy_threshold,
                                        value_allocator,
                                        vertex_size_allocator,
                                        vertex_allocator);

                std::int64_t community_count =
                    reindex_communities(labels, vertex_count, vertex_allocator);
                if (!changed) {
                    if (communities.empty()) {
                        labels_size.push_back(community_count);
                        vertex_size.push_back(vertex_count);
                        communities.push_back(labels);
                    }
                    else {
                        deallocate(vertex_allocator, labels, vertex_count);
                    }
                    break;
                }
                compress_graph(rows,
                               cols,
                               weights,
                               self_loops,
                               vertex_count,
                               community_count,
                               labels,
                               value_allocator,
                               vertex_allocator,
                               alloc_ptr);
                labels_size.push_back(community_count);
                vertex_size.push_back(vertex_count);
                communities.push_back(labels);
                vertex_count = community_count;
            }
            for (std::int64_t iteration = communities.size() - 2; iteration >= 0; --iteration) {
                // flat the communities from the next iteration
                for (std::int64_t v = 0; v < vertex_size[iteration]; ++v) {
                    communities[iteration][v] =
                        communities[iteration + 1][communities[iteration][v]];
                }
            }

            auto labels_arr = array<vertex_type>::empty(t.get_vertex_count());
            vertex_type* labels_ = labels_arr.get_mutable_data();
            for (std::int64_t v = 0; v < t.get_vertex_count(); ++v) {
                labels_[v] = communities[0][v];
            }

            deallocate(vertex_size_allocator, rows, t.get_vertex_count() + 1);
            deallocate(vertex_allocator, cols, edge_count * 2);
            deallocate(value_allocator, weights, edge_count * 2);
            deallocate(value_allocator, self_loops, t.get_vertex_count());
            for (int64_t iteration = 0; iteration < communities.size(); ++iteration) {
                deallocate(vertex_allocator, communities[iteration], labels_size[iteration]);
            }

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(labels_arr, t.get_vertex_count(), 1)
                                .build())
                .set_modularity(modularity)
                .set_community_count(vertex_count);
        }
    }
};

} // namespace oneapi::dal::preview::louvain::backend

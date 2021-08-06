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
#include <vector>
#include <algorithm>

#include <iostream>

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

inline void singleton_partition(std::vector<std::int32_t>& labels, std::int64_t vertex_count) {
    labels.resize(vertex_count);
    for (std::int64_t vertex = 0; vertex < vertex_count; ++vertex) {
        labels[vertex] = vertex;
    }
}

inline std::int64_t reindex_communities(std::vector<std::int32_t>& data) {
    std::unordered_map<std::int32_t, std::int32_t> index;
    std::int64_t count = 0;
    for (auto& value : data) {
        if (!index.count(value)) {
            index[value] = count++;
        }
        value = index[value];
    }
    return count;
}

template <typename EdgeValue>
inline void compress_graph(std::vector<std::int64_t>& rows,
                           std::vector<std::int32_t>& cols,
                           std::vector<EdgeValue>& vals,
                           std::vector<EdgeValue>& self_loops,
                           std::int64_t vertex_count,
                           std::vector<std::int32_t>& partition) {
    std::vector<std::unordered_map<std::int32_t, EdgeValue>> weights(vertex_count);
    std::vector<EdgeValue> c_self_loops(vertex_count);
    for (size_t vertex = 0; vertex < self_loops.size(); ++vertex) {
        std::int32_t c = partition[vertex];
        c_self_loops[c] += self_loops[vertex];
        for (std::int64_t index = rows[vertex]; index < rows[vertex + 1]; ++index) {
            std::int32_t to = cols[index];
            EdgeValue weight = vals[index];
            std::int32_t to_c = partition[to];
            if (c == to_c) {
                c_self_loops[c] += weight / 2;
            }
            else {
                weights[c][to_c] += weight;
            }
        }
    }
    rows.resize(vertex_count + 1);
    std::int64_t cols_size = 0;
    for (std::int32_t index = 0; index < vertex_count; ++index) {
        rows[index + 1] = rows[index] + weights[index].size();
        cols_size += weights[index].size();
    }
    cols.resize(cols_size);
    vals.resize(cols_size);
    for (std::int32_t index = 0, vertex = 0; vertex < vertex_count; ++vertex) {
        for (const auto& edge : weights[vertex]) {
            cols[index] = edge.first;
            vals[index++] = edge.second;
        }
    }
    self_loops = std::move(c_self_loops);
}

template <typename T>
inline double init_step(std::vector<std::int64_t>& rows,
                        std::vector<std::int32_t>& cols,
                        std::vector<T>& vals,
                        std::vector<T>& self_loops,
                        std::int64_t vertex_count,
                        std::vector<std::int32_t>& labels,
                        double resolution,
                        std::vector<T>& k,
                        std::vector<T>& tot,
                        T& m,
                        std::vector<std::int32_t>& community_size) {
    std::int32_t n_communities = 0;
    for (std::int32_t value : labels) {
        ++community_size[value];
        n_communities = std::max(n_communities, value);
    }
    ++n_communities;
    k.assign(vertex_count, 0);
    tot.assign(n_communities, 0);
    std::vector<T> k_c(n_communities);
    m = 0;
    std::vector<T> local_self_loops(n_communities);
    for (std::int64_t vertex = 0; vertex < vertex_count; ++vertex) {
        std::int32_t c = labels[vertex];
        local_self_loops[c] += self_loops[vertex];
        k_c[c] += self_loops[vertex] * 2;
        k[vertex] += self_loops[vertex] * 2;
        tot[c] += self_loops[vertex] * 2;
        m += self_loops[vertex];
        for (std::int64_t index = rows[vertex]; index < rows[vertex + 1]; ++index) {
            std::int32_t to = cols[index];
            T weight = vals[index];
            std::int32_t to_c = labels[to];
            k_c[c] += weight;
            k[vertex] += weight;
            tot[c] += weight;
            if (vertex < to) {
                m += weight;
                if (c == to_c) {
                    local_self_loops[c] += weight;
                }
            }
        }
    }

    double modularity = 0;
    for (std::int32_t community = 0; community < n_communities; ++community) {
        modularity += 1.0 / 2 / m *
                      (local_self_loops[community] * 2 -
                       resolution * k_c[community] * k_c[community] / 2.0 / m);
    }
    return modularity;
}

template <typename T>
inline double move_nodes(std::vector<std::int64_t>& rows,
                         std::vector<std::int32_t>& cols,
                         std::vector<T>& vals,
                         std::vector<T>& self_loops,
                         std::int64_t vertex_count,
                         std::vector<std::int32_t>& n2c,
                         bool& changed,
                         double resolution,
                         double accuracy_threshold) {
    std::vector<T> k(vertex_count);
    std::vector<T> tot(vertex_count);
    T m = 0;
    std::vector<std::int32_t> community_size(vertex_count);

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
                                  community_size);

    // interate over all vertices
    double old_modularity = modularity;
    std::vector<T> k_vertex_to(vertex_count);
    std::vector<T> neighboring_communities(vertex_count);
    std::vector<std::int32_t> random_order(vertex_count);
    for (size_t index = 0; index < random_order.size(); ++index) {
        random_order[index] = index;
    }
    //std::random_shuffle(random_order.begin(), random_order.end());
    std::vector<std::int32_t> empty_community(vertex_count);
    std::int32_t empty_count = 0;
    do {
        old_modularity = modularity;
        for (std::int32_t order_index = 0; order_index < vertex_count; ++order_index) {
            std::int32_t vertex = random_order[order_index];
            std::int32_t c_old = n2c[vertex];

            // calculate sum of weights of edges between vertex and community to move into
            std::int32_t community_count = 0;
            for (std::int32_t index = rows[vertex]; index < rows[vertex + 1]; ++index) {
                std::int32_t to = cols[index];
                std::int32_t community = n2c[to];
                T weight = vals[index];
                if (k_vertex_to[community] == 0) {
                    neighboring_communities[community_count++] = community;
                }
                k_vertex_to[community] += weight;
            }

            // remove vertex from the current community
            T k_iold = k_vertex_to[c_old];
            tot[c_old] -= k[vertex];
            double delta_modularity =
                1.0 * k_iold / m - resolution * tot[c_old] * k[vertex] / 2.0 / m / m;
            modularity -= delta_modularity;
            std::int32_t move_community = n2c[vertex];
            --community_size[c_old];
            if (!community_size[c_old]) {
                empty_community[empty_count++] = c_old;
            }
            // optionaly can be removed, but c_old community can be checked twice
            else if (empty_count) {
                neighboring_communities[community_count++] = empty_community[empty_count];
            }

            // iterate over nodes
            for (std::int32_t index = 0; index < community_count; ++index) {
                std::int32_t community = neighboring_communities[index];

                // try to move vertex to the community
                T k_ic = k_vertex_to[community];
                double delta =
                    1.0 * k_ic / m - resolution * tot[community] * k[vertex] / 2.0 / m / m;
                if (delta_modularity < delta) {
                    delta_modularity = delta;
                    move_community = community;
                }
                k_vertex_to[community] = 0;
            }
            k_vertex_to[c_old] = 0;

            // move vertex to the best community with the best modularity gain
            modularity += delta_modularity;
            double k_inew = k_vertex_to[move_community];
            tot[move_community] += k[vertex];
            n2c[vertex] = move_community;
            if (move_community != c_old) {
                changed = true;
            }
            ++community_size[move_community];
            if (community_size[move_community] == 1) {
                --empty_count;
            }
        }
    } while (modularity - old_modularity > accuracy_threshold);
    return modularity;
}

template <typename Cpu, typename EdgeValue>
struct louvain_kernel {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        const std::int32_t *init_partition,
        const EdgeValue *vals,
        byte_alloc_iface *alloc_ptr) {
        {
            using value_type = EdgeValue;
            using vertex_type = std::int32_t;
            using value_allocator_type = inner_alloc<value_type>;
            using vertex_allocator_type = inner_alloc<vertex_type>;

            vertex_allocator_type vertex_allocator(alloc_ptr);
            value_allocator_type value_allocator(alloc_ptr);

            double resolution = desc.get_resolution();
            double accuracy_threshold = desc.get_accuracy_threshold();
            std::int64_t max_iteration_count = desc.get_max_iteration_count();

            auto vertex_count = t.get_vertex_count();
            auto edge_count = t.get_edge_count();
            std::vector<std::int64_t> rows(t._rows_ptr, t._rows_ptr + vertex_count + 1);
            std::vector<vertex_type> cols(t._cols_ptr, t._cols_ptr + edge_count * 2);
            std::vector<EdgeValue> weights(vals, vals + edge_count * 2);
            std::vector<EdgeValue> self_loops(vertex_count);

            std::vector<std::int32_t> labels;
            double modularity = -1;
            if (init_partition != nullptr) {
                labels.resize(vertex_count);
                for (size_t v = 0; v < labels.size(); ++v) {
                    labels[v] = init_partition[v];
                }
            }
            else {
                singleton_partition(labels, vertex_count);
            }
            std::vector<std::vector<std::int32_t>> communities;
            for (std::int64_t iteration = 0;
                 iteration < max_iteration_count || !max_iteration_count;
                 ++iteration, singleton_partition(labels, vertex_count)) {
                bool changed = false;
                modularity = move_nodes(rows,
                                        cols,
                                        weights,
                                        self_loops,
                                        vertex_count,
                                        labels,
                                        changed,
                                        resolution,
                                        accuracy_threshold);

                vertex_count = reindex_communities(labels);
                if (!changed) {
                    if (communities.empty()) {
                        communities.push_back(std::move(labels));
                    }
                    break;
                }
                compress_graph<EdgeValue>(rows, cols, weights, self_loops, vertex_count, labels);
                communities.push_back(std::move(labels));
            }
            for (std::int32_t iteration = static_cast<std::int32_t>(communities.size()) - 2;
                 iteration >= 0;
                 --iteration) {
                // flat the communities from the next iteration
                for (size_t v = 0; v < communities[iteration].size(); ++v) {
                    communities[iteration][v] =
                        communities[iteration + 1][communities[iteration][v]];
                }
            }

            auto labels_arr = array<vertex_type>::empty(t.get_vertex_count());
            vertex_type* labels_ = labels_arr.get_mutable_data();
            for (std::int64_t i = 0; i < t.get_vertex_count(); ++i) {
                labels_[i] = communities[0][i];
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

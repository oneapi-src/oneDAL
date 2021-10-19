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

#include "oneapi/dal/algo/connected_components/common.hpp"
#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::preview::connected_components::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

//Given an edge (u, v), ensures that u and v are within the same component tree, or connects them otherwise
template <typename Cpu>
void link(const std::int32_t &u, const std::int32_t &v, std::int32_t *components) {
    std::int32_t p1 = components[u];
    std::int32_t p2 = components[v];
    std::int32_t h = 0;
    std::int32_t l = 0;
    while (p1 != p2) {
        if (p1 > p2) {
            h = p1;
            l = p2;
        }
        else {
            h = p2;
            l = p1;
        }
        if (components[h] == h) {
            components[h] = l;
            break;
        }
        p1 = components[components[h]];
        p2 = components[l];
    }
}

//Reduces component trees to single-level depth
template <typename Cpu>
void compress(const std::int32_t &u, std::int32_t *components) {
    while (components[components[u]] != components[u]) {
        components[u] = components[components[u]];
    }
}

template <typename Cpu>
void order_component_ids(const std::int64_t &vertex_count,
                         std::int64_t &component_count,
                         std::int32_t *components) {
    std::int32_t ordered_comp_id = 0;

    for (std::int32_t u = 0; u < vertex_count; ++u) {
        if (components[u] == u) {
            components[u] = ordered_comp_id;
            component_count++;
            ordered_comp_id++;
        }
        else {
            components[u] = components[components[u]];
        }
    }
}

template <typename Cpu>
std::int32_t most_frequent_element(const std::int32_t *components,
                                   const std::int64_t &vertex_count,
                                   inner_alloc<std::int32_t> &vertex_allocator,
                                   const std::int64_t &samples_count = 1024) {
    std::int32_t *rnd_vertex_ids = allocate(vertex_allocator, samples_count);

    dal::backend::primitives::engine eng;
    dal::backend::primitives::rng<std::int32_t> rn_gen;
    rn_gen.uniform(samples_count, rnd_vertex_ids, eng.get_state(), 0, vertex_count);

    std::int32_t *root_sample_counts = allocate(vertex_allocator, vertex_count);

    for (std::int32_t u = 0; u < vertex_count; ++u) {
        root_sample_counts[u] = 0;
    }
    for (std::int32_t i = 0; i < samples_count; ++i) {
        root_sample_counts[components[rnd_vertex_ids[i]]]++;
    }
    deallocate(vertex_allocator, rnd_vertex_ids, samples_count);

    std::int32_t max_root_sample_count = 0;
    std::int32_t most_frequent_root = 0;
    for (std::int32_t u = 0; u < vertex_count; ++u) {
        if (root_sample_counts[u] > max_root_sample_count) {
            max_root_sample_count = root_sample_counts[u];
            most_frequent_root = u;
        }
    }
    deallocate(vertex_allocator, root_sample_counts, vertex_count);

    return most_frequent_root;
}

template <typename Cpu>
struct afforest {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        byte_alloc_iface *alloc_ptr) {
        using vertex_allocator_type = inner_alloc<std::int32_t>;
        vertex_allocator_type vertex_allocator(alloc_ptr);
        const auto vertex_count = t.get_vertex_count();

        std::int32_t *components = allocate(vertex_allocator, vertex_count);
        for (std::int32_t u = 0; u < vertex_count; ++u) {
            components[u] = u;
        }

        const std::int32_t neighbors_round = 2;

        for (std::int32_t i = 0; i < neighbors_round; ++i) {
            for (std::int32_t u = 0; u < vertex_count; ++u) {
                std::int32_t neighbors_count = t.get_vertex_degree(u);
                if (i < neighbors_count) {
                    link<Cpu>(u, t.get_vertex_neighbors_begin(u)[i], components);
                }
            }
            for (std::int32_t u = 0; u < vertex_count; ++u) {
                compress<Cpu>(u, components);
            }
        }

        const std::int32_t sample_comp =
            most_frequent_element<Cpu>(components, vertex_count, vertex_allocator);

        for (std::int32_t u = 0; u < vertex_count; ++u) {
            if (components[u] != sample_comp) {
                if (t.get_vertex_degree(u) >= neighbors_round) {
                    for (auto v = t.get_vertex_neighbors_begin(u);
                         v != t.get_vertex_neighbors_end(u);
                         ++v) {
                        link<Cpu>(u, *v, components);
                    }
                }
            }
        }

        for (std::int32_t u = 0; u < vertex_count; ++u) {
            compress<Cpu>(u, components);
        }

        std::int64_t component_count = 0;
        order_component_ids<Cpu>(vertex_count, component_count, components);

        auto labels_arr = array<std::int32_t>::empty(vertex_count);
        std::int32_t *labels = labels_arr.get_mutable_data();
        dal::backend::copy<std::int32_t>(labels, components, vertex_count);
        deallocate(vertex_allocator, components, vertex_count);

        return vertex_partitioning_result<task::vertex_partitioning>()
            .set_labels(homogen_table::wrap(labels_arr, vertex_count, 1))
            .set_component_count(component_count);
    }
};

} // namespace oneapi::dal::preview::connected_components::backend

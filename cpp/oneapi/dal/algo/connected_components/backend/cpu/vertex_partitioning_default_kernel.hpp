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

inline void link(const std::int32_t &u, const std::int32_t &v, std::int32_t *components) {
    std::int32_t p1 = components[u];
    std::int32_t p2 = components[v];
    std::int32_t h;
    std::int32_t l;
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

inline void compress(const std::int32_t &u, std::int32_t *components) {
    while (components[components[u]] != components[u]) {
        components[u] = components[components[u]];
    }
}

inline void order_component_ids(const std::int64_t &vertex_count,
                                std::int64_t &component_count,
                                std::int32_t *components) {
    std::int32_t ordered_comp_id = 0;
    const auto max_vertex_id = dal::detail::integral_cast<std::int32_t>(vertex_count - 1);
    for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
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
inline std::int32_t most_frequent_element(const std::int32_t *components,
                                          const std::int64_t &vertex_count,
                                          byte_alloc_iface *alloc_ptr,
                                          const std::int32_t &samples_num = 1024) {
    using vertex_type = std::int32_t;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);

    vertex_type *rnd_vertex_ids = allocate(vertex_allocator, samples_num);

    dal::backend::primitives::engine eng;
    dal::backend::primitives::rng<std::int32_t> rn_gen;
    rn_gen.uniform(samples_num, rnd_vertex_ids, eng.get_state(), 0, vertex_count);

    vertex_type *sample_counts = allocate(vertex_allocator, vertex_count);
    const auto max_vertex_id = dal::detail::integral_cast<std::int32_t>(vertex_count - 1);
    for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
        sample_counts[u] = 0;
    }
    for (std::int32_t i = 0; i < samples_num; ++i) {
        sample_counts[components[rnd_vertex_ids[i]]]++;
    }
    deallocate(vertex_allocator, rnd_vertex_ids, samples_num);

    std::int32_t max_sample_count = 0;
    std::int32_t most_frequent_root = 0;
    for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
        if (sample_counts[u] > max_sample_count) {
            max_sample_count = sample_counts[u];
            most_frequent_root = u;
        }
    }
    deallocate(vertex_allocator, sample_counts, vertex_count);

    return most_frequent_root;
}

template <typename Cpu>
struct afforest {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        byte_alloc_iface *alloc_ptr) {
        using vertex_type = std::int32_t;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);

        const auto vertex_count = t.get_vertex_count();
        const auto max_vertex_id = dal::detail::integral_cast<std::int32_t>(vertex_count - 1);

        vertex_type *components = allocate(vertex_allocator, vertex_count);

        for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
            components[u] = u;
        }

        const std::int32_t neighbors_round = 2;

        for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
            std::int32_t neighbors_count = t.get_vertex_degree(u);
            for (std::int32_t v = 0; (v < neighbors_count) && (v < neighbors_round); ++v) {
                link(u, t.get_vertex_neighbors_begin(u)[v], components);
            }
        }

        const std::int32_t sample_comp =
            most_frequent_element<Cpu>(components, vertex_count, alloc_ptr);

        for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
            if (components[u] != sample_comp) {
                if (t.get_vertex_degree(u) >= neighbors_round) {
                    for (auto v = t.get_vertex_neighbors_begin(u);
                         v != t.get_vertex_neighbors_end(u);
                         ++v) {
                        link(u, *v, components);
                    }
                }
            }
        }

        for (std::int32_t u = 0; u <= max_vertex_id; ++u) {
            compress(u, components);
        }

        std::int64_t component_count = 0;
        order_component_ids(vertex_count, component_count, components);

        auto labels_arr = array<vertex_type>::empty(vertex_count);
        vertex_type *labels = labels_arr.get_mutable_data();
        dal::backend::copy<vertex_type>(labels, components, vertex_count);
        deallocate(vertex_allocator, components, vertex_count);

        return vertex_partitioning_result<task::vertex_partitioning>()
            .set_labels(homogen_table::wrap(labels_arr, vertex_count, 1))
            .set_component_count(component_count);
    }
};

} // namespace oneapi::dal::preview::connected_components::backend

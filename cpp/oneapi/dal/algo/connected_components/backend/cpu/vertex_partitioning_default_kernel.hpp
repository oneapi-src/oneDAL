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
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::preview::connected_components::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

inline void link(std::int64_t u, std::int64_t v, std::int32_t *components) {
    std::int32_t p1;
    std::int32_t p2;
    std::int32_t h;
    std::int32_t l;
    p1 = components[u];
    p2 = components[v];
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

inline void compress(std::int64_t u, std::int32_t *components) {
    while (components[components[u]] != components[u]) {
        components[u] = components[components[u]];
    }
}

inline void order_component_ids(const std::int64_t vertex_count,
                                std::int64_t &component_count,
                                std::int32_t *components) {
    std::int64_t ordered_comp_id = 0;

    for (auto i = 0; i < vertex_count; ++i) {
        if (components[i] == i) {
            components[i] = ordered_comp_id;
            component_count++;
            ordered_comp_id++;
        }
        else {
            components[i] = components[components[i]];
        }
    }
}

template <typename Cpu>
inline std::int32_t most_frequent_element(const std::int32_t *components,
                                          const std::int64_t vertex_count,
                                          byte_alloc_iface *alloc_ptr,
                                          const std::int64_t samples_num = 1024) {
    using vertex_type = std::int32_t;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);

    vertex_type *sample_counts = allocate(vertex_allocator, vertex_count);
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        sample_counts[i] = 0;
    }

    vertex_type *rnd_vertex_ids = allocate(vertex_allocator, samples_num);

    dal::backend::primitives::engine eng;
    dal::backend::primitives::rng<std::int32_t> rn_gen;

    rn_gen.uniform(samples_num, rnd_vertex_ids, eng.get_state(), 0, vertex_count);

    for (std::int64_t i = 0; i < samples_num; i++) {
        sample_counts[components[rnd_vertex_ids[i]]]++;
    }
    deallocate(vertex_allocator, rnd_vertex_ids, samples_num);

    std::int64_t max_sample_count = 0;
    std::int64_t most_frequent_root = 0;
    for (std::int64_t i = 0; i < vertex_count; i++) {
        if (sample_counts[i] > max_sample_count) {
            max_sample_count = sample_counts[i];
            most_frequent_root = i;
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

        vertex_type *components = allocate(vertex_allocator, vertex_count);
        for (std::int64_t i = 0; i < vertex_count; ++i) {
            components[i] = i;
        }

        const std::int32_t neighbors_round = 2;

        for (std::int64_t i = 0; i < vertex_count; ++i) {
            std::int32_t neighbors_count = t.get_vertex_degree(i);
            for (std::int32_t j = 0; (j < neighbors_count) && (j < neighbors_round); ++j) {
                link(i, t.get_vertex_neighbors_begin(i)[j], components);
            }
        }

        const std::int32_t sample_comp =
            most_frequent_element<Cpu>(components, vertex_count, alloc_ptr);

        for (std::int64_t i = 0; i < vertex_count; ++i) {
            if (components[i] != sample_comp) {
                if (t.get_vertex_degree(i) >= neighbors_round) {
                    for (auto j = t.get_vertex_neighbors_begin(i);
                         j != t.get_vertex_neighbors_end(i);
                         ++j) {
                        link(i, *j, components);
                    }
                }
            }
        }

        for (std::int64_t i = 0; i < vertex_count; ++i) {
            compress(i, components);
        }

        std::int64_t component_count = 0;
        order_component_ids(vertex_count, component_count, components);

        auto labels_arr = array<vertex_type>::empty(vertex_count);
        vertex_type *labels = labels_arr.get_mutable_data();
        dal::backend::copy<vertex_type>(labels, components, vertex_count);
        deallocate(vertex_allocator, components, vertex_count);

        return vertex_partitioning_result<task::vertex_partitioning>()
            .set_labels(homogen_table::wrap(labels_arr, t.get_vertex_count(), 1))
            .set_component_count(component_count);
    }
};

} // namespace oneapi::dal::preview::connected_components::backend

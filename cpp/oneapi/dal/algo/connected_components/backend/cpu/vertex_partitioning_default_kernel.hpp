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
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/algo/connected_components/backend/cpu/vertex_partitioning_rng.hpp"

namespace oneapi::dal::preview::connected_components::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename T>
inline bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
    if (x == old_val) {
        x = new_val;
        return true;
    }
    return false;
}

inline void link(std::int64_t u, std::int64_t v, std::int64_t *D) {
    std::int64_t p1;
    std::int64_t p2;
    std::int64_t h;
    std::int64_t l;
    p1 = D[u];
    p2 = D[v];
    while (p1 != p2) {
        if (p1 > p2) {
            h = p1;
            l = p2;
        }
        else {
            h = p2;
            l = p1;
        }
        if (compare_and_swap<std::int64_t>(D[h], h, l)) {
            break;
        }
        p1 = D[D[h]];
        p2 = D[l];
    }
}

inline void compress(std::int64_t u, std::int64_t *D) {
    while (D[D[u]] != D[u]) {
        D[u] = D[D[u]];
    }
}

inline void order_component_ids(const std::int64_t &vertex_count,
                                std::int64_t &component_count,
                                std::int64_t *D) {
    std::int64_t ordered_comp_id = 0;
    component_count = 0;

    for (auto i = 0; i < vertex_count; ++i) {
        if (D[i] == i) {
            D[i] = ordered_comp_id;
            component_count++;
            ordered_comp_id++;
        }
        else {
            D[i] = D[D[i]];
        }
    }
}

template <typename Cpu>
inline std::int32_t most_frequent_element(std::int64_t *D,
                                          std::int64_t vertex_count,
                                          byte_alloc_iface *alloc_ptr,
                                          std::int32_t samples_num = 1024) {
    using vertex_type = std::int64_t;
    using vertex_allocator_type = inner_alloc<vertex_type>;

    vertex_allocator_type vertex_allocator(alloc_ptr);

    vertex_type *samples = allocate(vertex_allocator, vertex_count);
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        samples[i] = 0;
    }

    rnd_seq<Cpu, std::int64_t> gen(samples_num, 0, vertex_count);
    auto uniform_values = gen.get_data();

    for (std::int64_t i = 0; i < samples_num; i++) {
        samples[D[uniform_values[i]]]++;
    }
    std::int64_t max_sample = 0;
    std::int64_t max_sample_root = 0;
    for (std::int64_t i = 0; i < vertex_count; i++) {
        if (samples[i] > max_sample) {
            max_sample = samples[i];
            max_sample_root = i;
        }
    }

    return max_sample_root;
}

template <typename Cpu>
struct afforest {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t,
        byte_alloc_iface *alloc_ptr) {
        {
            using vertex_type = std::int64_t;
            using vertex_allocator_type = inner_alloc<vertex_type>;

            vertex_allocator_type vertex_allocator(alloc_ptr);

            const auto vertex_count = t.get_vertex_count();

            vertex_type *D = allocate(vertex_allocator, vertex_count);
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                D[i] = i;
            }

            std::int32_t neighbors_round = 2;

            for (std::int64_t i = 0; i < vertex_count; ++i) {
                std::int32_t neighbors_count = t.get_vertex_degree(i);
                for (std::int32_t j = 0; (j < neighbors_count) && (j < neighbors_round); ++j) {
                    link(i, t.get_vertex_neighbors_begin(i)[j], D);
                }
            }

            std::int32_t sample_comp = most_frequent_element<Cpu>(D, vertex_count, alloc_ptr);

            for (std::int64_t i = 0; i < vertex_count; ++i) {
                if (D[i] != sample_comp) {
                    if (t.get_vertex_degree(i) >= neighbors_round) {
                        for (auto j = t.get_vertex_neighbors_begin(i);
                             j != t.get_vertex_neighbors_end(i);
                             ++j) {
                            link(i, *j, D);
                        }
                    }
                }
            }

            for (std::int64_t i = 0; i < vertex_count; ++i) {
                compress(i, D);
            }

            std::int64_t component_count = 0;
            order_component_ids(vertex_count, component_count, D);

            auto label_arr = array<vertex_type>::empty(vertex_count);
            vertex_type *label_ = label_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                label_[i] = D[i];
            }
            deallocate(vertex_allocator, D, vertex_count);

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(label_arr, t.get_vertex_count(), 1)
                                .build())
                .set_component_count(component_count);
        }
    }
};

} // namespace oneapi::dal::preview::connected_components::backend

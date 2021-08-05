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

#include <random>
#include <unordered_map>
#include <algorithm>

#include "oneapi/dal/algo/connected_components/common.hpp"
#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

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

inline void link(std::int64_t u, std::int64_t v, vector<int64_t> &D) {
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

inline void compress(std::int64_t u, std::vector<std::int64_t> &D) {
    while (D[D[u]] != D[u]) {
        D[u] = D[D[u]];
    }
}

inline void order_component_ids(const std::int64_t &vertex_count,
                                std::int64_t &component_count,
                                std::vector<std::int64_t> &D) {
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

inline bool compare_sample_counts(std::unordered_map<std::int64_t, std::int64_t>::value_type &a,
                               std::unordered_map<std::int64_t, std::int64_t>::value_type &b) {
    return (a.second < b.second);
}

inline std::int32_t most_frequent_element(vector<int64_t> &D, std::int32_t samples_num = 10) {
    std::default_random_engine generator;
    std::uniform_int_distribution<std::int64_t> distribution(0, D.size() - 1);

    std::unordered_map<std::int64_t, std::int64_t> sample_counts;

    for (std::int32_t i = 0; i < samples_num; ++i) {
        std::int64_t vertex = distribution(generator);
        sample_counts[D[vertex]]++;
    }
    auto sample_component =
        std::max_element(sample_counts.begin(), sample_counts.end(), compare_sample_counts);

    return sample_component->first;
}

template <typename Cpu>
struct afforest {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning> &desc,
        const dal::preview::detail::topology<std::int32_t> &t) {
        {
            using value_type = std::int32_t;
            const auto vertex_count = t.get_vertex_count();

            std::vector<std::int64_t> D;
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                D.push_back(i);
            }

            std::int32_t neighbors_round = 2;

            for (std::int64_t i = 0; i < vertex_count; ++i) {
                std::int32_t neighbors_count = t.get_vertex_degree(i);
                for (std::int32_t j = 0; (j < neighbors_count) && (j < neighbors_round); ++j) {
                    link(i, t.get_vertex_neighbors_begin(i)[j], D);
                }
            }

            std::int32_t samples_num = 10;
            std::int32_t sample_comp = most_frequent_element(D, samples_num);

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

            auto label_arr = array<value_type>::empty(vertex_count);
            value_type *label_ = label_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                label_[i] = D[i];
            }

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(label_arr, t.get_vertex_count(), 1)
                                .build())
                .set_component_count(component_count);
        }
    }
};

} // namespace oneapi::dal::preview::connected_components::backend

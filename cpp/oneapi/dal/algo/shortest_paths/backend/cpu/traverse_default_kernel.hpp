
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

#include <atomic>

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

namespace mode {
struct distances;
struct distances_predecessors;
} // namespace mode

template <typename EV, typename VT>
struct dist_pred {
    using distance_type = EV;
    using vertex_type = VT;
    dist_pred(const EV& dist_, const VT& pred_) : dist(dist_), pred(pred_) {}
    EV dist = 0;
    VT pred = 0;
};

template <typename T1, typename T2, typename T1_, typename T2_>
inline constexpr bool is_class_params_the_same_v =
    std::conjunction_v<std::is_same<T1, T2>, std::is_same<T1, T1_>, std::is_same<T2, T2_>>;

template <typename T1, typename T2, typename T1_, typename T2_>
inline constexpr bool is_not_class_params_the_same_v =
    !std::is_same_v<T1, T2> && std::is_same_v<T1, T1_> && std::is_same_v<T2, T2_>;

template <typename Cpu, typename Mode, typename Vertex, typename Value, typename AtomicT>
class data_to_relax_base {
public:
    using relaxing_data_type = Value;
    using atomic_type = std::atomic<AtomicT>;
    using atomic_value_type = AtomicT;
    using atomic_value_allocator_type = inner_alloc<atomic_type>;
    using vertex_type = Vertex;
    using mode_type = Mode;

    template <typename M = Mode, std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    data_to_relax_base(std::int64_t vertex_count,
                       Vertex source,
                       relaxing_data_type max_dist,
                       byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              atomic_value_allocator(alloc_ptr) {
        distances = allocate(atomic_value_allocator, vertex_count);
        distances = new (distances) atomic_type[vertex_count]();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            store(i, max_dist);
        });
        store(source, 0);
    }

    template <typename EV,
              typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    data_to_relax_base(std::int64_t vertex_count,
                       Vertex source,
                       EV max_dist,
                       byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              atomic_value_allocator(alloc_ptr) {
        distances = allocate(atomic_value_allocator, vertex_count);
        //distances = new (distances) atomic_type[vertex_count];
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            new (distances + i) relaxing_data_type(max_dist, -1);
        });
        store(source, relaxing_data_type(0, -1));
    }

    virtual ~data_to_relax_base() {
        deallocate(atomic_value_allocator, distances, vertex_count);
    }

    inline atomic_type* get_distances_ptr() {
        return distances;
    }

    template <
        typename V = relaxing_data_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type>,
                         bool> = true>
    inline relaxing_data_type load(Vertex u) const {
        return distances[u].load();
    }

    template <
        typename V = relaxing_data_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type>,
                         bool> = true>
    inline void store(Vertex u, relaxing_data_type value) {
        distances[u].store(value);
    }

    template <
        typename V = relaxing_data_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type>,
                         bool> = true>
    inline bool compare_exchange_strong(Vertex u,
                                        relaxing_data_type& old_value,
                                        relaxing_data_type new_value) {
        return distances[u].compare_exchange_strong(old_value, new_value);
    }

    template <typename V = relaxing_data_type,
              typename AV = atomic_value_type,
              std::enable_if_t<
                  is_not_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type> &&
                      std::is_same_v<relaxing_data_type, double> &&
                      std::is_same_v<atomic_value_type, std::int64_t>,
                  bool> = true>
    inline double load(Vertex u) const {
        std::int64_t a_int = distances[u].load();
        std::int64_t* a_int_ptr = &a_int;
        return *reinterpret_cast<double*>(a_int_ptr);
    }

    template <typename V = relaxing_data_type,
              typename AV = atomic_value_type,
              std::enable_if_t<
                  is_not_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type> &&
                      std::is_same_v<relaxing_data_type, double> &&
                      std::is_same_v<atomic_value_type, std::int64_t>,
                  bool> = true>
    inline void store(Vertex u, double value) {
        double* value_ptr = &value;
        std::int64_t value_int_representation = *reinterpret_cast<std::int64_t*>(value_ptr);
        distances[u].store(value_int_representation);
    }

    template <typename V = relaxing_data_type,
              typename AV = atomic_value_type,
              std::enable_if_t<
                  is_not_class_params_the_same_v<V, AV, relaxing_data_type, atomic_value_type> &&
                      std::is_same_v<relaxing_data_type, double> &&
                      std::is_same_v<atomic_value_type, std::int64_t>,
                  bool> = true>
    inline bool compare_exchange_strong(Vertex u, double& old_value, double new_value) {
        double* old_value_ptr = &old_value;
        double* new_value_ptr = &new_value;
        std::int64_t old_value_int_representation = *reinterpret_cast<std::int64_t*>(old_value_ptr);
        std::int64_t new_value_int_representation = *reinterpret_cast<std::int64_t*>(new_value_ptr);
        return distances[u].compare_exchange_strong(old_value_int_representation,
                                                    new_value_int_representation);
    }

    inline relaxing_data_type operator[](Vertex u) const {
        return load(u);
    }

    template <typename M = Mode, std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    inline auto get_distance(Vertex u) const {
        return load(u);
    }

    template <typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    inline auto get_distance(Vertex u) const {
        return load(u).dist;
    }

    template <typename M = Mode, std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    inline auto get_predecessor(Vertex u) const {
        return -1;
    }

    template <typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    inline auto get_predecessor(Vertex u) const {
        return load(u).pred;
    }

private:
    data_to_relax_base(const data_to_relax_base&) = delete;
    data_to_relax_base(data_to_relax_base&&) = delete;
    const std::int64_t vertex_count;
    atomic_value_allocator_type atomic_value_allocator;
    atomic_type* distances;
};

template <typename Cpu, typename Mode, typename Vertex, typename Value>
class data_to_relax : public data_to_relax_base<Cpu, Mode, Vertex, Value, Value> {
    using data_to_relax_base<Cpu, Mode, Vertex, Value, Value>::data_to_relax_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Cpu, typename Mode, typename Vertex>
class data_to_relax<Cpu, Mode, Vertex, double>
        : public data_to_relax_base<Cpu, Mode, Vertex, double, std::int64_t> {
    using data_to_relax_base<Cpu, Mode, Vertex, double, std::int64_t>::data_to_relax_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Vertex, typename EdgeValue, typename BinsVector>
inline void update_bins(const Vertex& v,
                        const EdgeValue& new_dist,
                        const EdgeValue& delta,
                        BinsVector& local_bins) {
    ONEDAL_ASSERT(new_dist > 0);
    ONEDAL_ASSERT(delta > 0);
    ONEDAL_ASSERT(new_dist / delta <= std::numeric_limits<EdgeValue>::max());
    const std::int64_t dest_bin = static_cast<std::int64_t>(new_dist / delta);
    ONEDAL_ASSERT(dest_bin >= 0);
    if (dest_bin >= local_bins.size()) {
        local_bins.resize(dest_bin + 1);
    }
    local_bins[dest_bin].push_back(v);
}

template <typename Mode>
struct relax_edges {};

template <>
struct relax_edges<mode::distances> {
    template <typename Topology, typename EdgeValue, typename BinsVector, typename DataToRelax>
    inline void operator()(const Topology& t,
                           const EdgeValue* vals,
                           typename Topology::vertex_type u,
                           EdgeValue delta,
                           DataToRelax& dist,
                           BinsVector& local_bins) {
        for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
            const auto v = t._cols_ptr[v_];
            const auto v_w = vals[v_];
            EdgeValue old_dist = dist[v];
            const EdgeValue new_dist = dist[u] + v_w;
            while (new_dist < old_dist) {
                if (dist.compare_exchange_strong(v, old_dist, new_dist)) {
                    update_bins(v, new_dist, delta, local_bins);
                    break;
                }
                old_dist = dist[v];
            }
        }
    }
};

template <>
struct relax_edges<mode::distances_predecessors> {
    template <typename Topology, typename EdgeValue, typename BinsVector, typename DataToRelax>
    inline void operator()(const Topology& t,
                           const EdgeValue* vals,
                           typename Topology::vertex_type u,
                           EdgeValue delta,
                           DataToRelax& dp,
                           BinsVector& local_bins) {
        for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
            const auto v = t._cols_ptr[v_];
            const auto v_w = vals[v_];
            auto old_dp = dp[v];
            EdgeValue old_dist = old_dp.dist;
            const EdgeValue new_dist = dp[u].dist + v_w;
            while (new_dist < old_dist) {
                if (dp.compare_exchange_strong(
                        v,
                        old_dp,
                        dist_pred<EdgeValue, typename Topology::vertex_type>(new_dist, u))) {
                    update_bins(v, new_dist, delta, local_bins);
                    break;
                }
                old_dp = dp[v];
                old_dist = old_dp.dist;
            }
        }
    }
};

template <typename BinsVector>
inline void find_next_bin_index(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    std::atomic<std::int64_t> min_nonempty_bucket_index;
    min_nonempty_bucket_index.store(max_bin_count);

    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        const auto& local_bin = local_bins[i];
        for (std::int64_t bucket_index = 0; bucket_index < local_bin.size(); bucket_index++) {
            if (!local_bin[bucket_index].empty()) {
                const std::int64_t new_min_index =
                    std::min(min_nonempty_bucket_index.load(), bucket_index);
                std::int64_t old_min_index = min_nonempty_bucket_index.load();
                while (new_min_index < old_min_index) {
                    if (min_nonempty_bucket_index.compare_exchange_strong(old_min_index,
                                                                          new_min_index)) {
                        break;
                    }
                    old_min_index = min_nonempty_bucket_index.load();
                }
                break;
            }
        }
    });

    curr_bin_index = min_nonempty_bucket_index.load();
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin(const std::int64_t& curr_bin_index,
                                         BinsVector& local_bins,
                                         SharedBinContainer& shared_bin) {
    std::atomic<std::int64_t> vertex_count_in_shared_bin = 0;
    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        auto& local_bin = local_bins[i];
        if (curr_bin_index < local_bin.size()) {
            auto& bucket = local_bin[curr_bin_index];
            if (bucket.size() != 0) {
                std::int64_t copy_start = vertex_count_in_shared_bin.fetch_add(bucket.size());
                copy(bucket.begin(), bucket.end(), shared_bin.get_mutable_data() + copy_start);
                bucket.resize(0);
            }
        }
    });
    return vertex_count_in_shared_bin.load();
}

template <typename Mode, typename Vertex, typename Value>
struct relaxing_data_type {};

template <typename Vertex, typename Value>
struct relaxing_data_type<mode::distances, Vertex, Value> {
    using value_type = Value;
};

template <typename Vertex, typename Value>
struct relaxing_data_type<mode::distances_predecessors, Vertex, Value> {
    using value_type = dist_pred<Value, Vertex>;
};

template <typename Cpu, typename EdgeValue, typename Mode>
struct delta_stepping {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        using value_type = EdgeValue;
        using vertex_type = std::int32_t;
        using value_allocator_type = inner_alloc<value_type>;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);
        value_allocator_type value_allocator(alloc_ptr);

        const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());
        const value_type delta = desc.get_delta();

        const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t max_elements_in_bin = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();
        using relaxing_data_t =
            typename relaxing_data_type<Mode, vertex_type, value_type>::value_type;
        data_to_relax<Cpu, Mode, vertex_type, relaxing_data_t> dist(vertex_count,
                                                                    source,
                                                                    max_dist,
                                                                    alloc_ptr);

        vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                        vertex_allocator);

        shared_bin[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t vertex_count_in_shared_bin = 1;
        std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

        using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
        using v1a_t = inner_alloc<v1v_t>;
        using v2v_t = vector_container<v1v_t, v1a_t>;
        using v2a_t = inner_alloc<v2v_t>;
        using v3v_t = vector_container<v2v_t, v2a_t>;
        v2a_t v2a(alloc_ptr);
        v3v_t local_bins(thread_cnt, v2a);
        local_bins[0].reserve(t.get_vertex_degree(source));

        vertex_type* local_processing_bins =
            allocate(vertex_allocator, max_elements_in_bin * thread_cnt);
        while (curr_bin_index != max_bin_count) {
            dal::detail::threader_for(
                vertex_count_in_shared_bin,
                vertex_count_in_shared_bin,
                [&](std::int64_t i) {
                    const vertex_type u = shared_bin[i];
                    if (dist.get_distance(u) >= delta * static_cast<value_type>(curr_bin_index)) {
                        relax_edges<Mode>()(
                            t,
                            vals,
                            u,
                            delta,
                            dist,
                            local_bins[dal::detail::threader_get_current_thread_index()]);
                    }
                });

            dal::detail::threader_for(thread_cnt, thread_cnt, [&](std::int64_t i) {
                const std::int64_t thread_id = i;
                auto& local_bin = local_bins[thread_id];
                while (curr_bin_index < local_bin.size() && !local_bin[curr_bin_index].empty() &&
                       local_bin[curr_bin_index].size() < max_elements_in_bin) {
                    auto copy_begin = local_processing_bins + max_elements_in_bin * thread_id;
                    auto& bucket = local_bin[curr_bin_index];
                    copy(bucket.begin(), bucket.end(), copy_begin);
                    const std::int64_t copy_count = bucket.size();
                    bucket.resize(0);
                    for (std::int64_t j = 0; j < copy_count; ++j) {
                        relax_edges<Mode>()(t, vals, copy_begin[j], delta, dist, local_bin);
                    }
                }
            });

            find_next_bin_index(curr_bin_index, local_bins);

            vertex_count_in_shared_bin =
                reduce_to_common_bin(curr_bin_index, local_bins, shared_bin);
        }

        deallocate(vertex_allocator, local_processing_bins, max_elements_in_bin * thread_cnt);

        if ((desc.get_optional_results() & optional_results::distances) &&
            (desc.get_optional_results() & optional_results::predecessors)) {
            auto dist_arr = array<value_type>::empty(vertex_count);
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();
            vertex_type* pred_ = pred_arr.get_mutable_data();

            dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
                dist_[i] = dist.get_distance(i);
                pred_[i] = dist.get_predecessor(i);
            });

            return traverse_result<task::one_to_all>()
                .set_distances(dal::detail::homogen_table_builder{}
                                   .reset(dist_arr, t.get_vertex_count(), 1)
                                   .build())
                .set_predecessors(dal::detail::homogen_table_builder{}
                                      .reset(pred_arr, t.get_vertex_count(), 1)
                                      .build());
        }
        else if (desc.get_optional_results() & optional_results::predecessors) {
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* pred_ = pred_arr.get_mutable_data();

            dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
                pred_[i] = dist.get_predecessor(i);
            });

            return traverse_result<task::one_to_all>().set_predecessors(
                dal::detail::homogen_table_builder{}
                    .reset(pred_arr, t.get_vertex_count(), 1)
                    .build());
        }
        else { // if(desc.get_optional_results() & optional_results::distances)
            auto dist_arr = array<value_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();

            dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
                dist_[i] = dist.get_distance(i);
            });

            return traverse_result<task::one_to_all>().set_distances(
                dal::detail::homogen_table_builder{}
                    .reset(dist_arr, t.get_vertex_count(), 1)
                    .build());
        }
    }
};

} // namespace oneapi::dal::preview::shortest_paths::backend


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

template <typename T1, typename T2, typename T1_, typename T2_>
inline constexpr bool is_class_params_the_same_v =
    std::conjunction_v<std::is_same<T1, T2>, std::is_same<T1, T1_>, std::is_same<T2, T2_>>;

template <typename T1, typename T2, typename T1_, typename T2_>
inline constexpr bool is_not_class_params_the_same_v =
    !std::is_same_v<T1, T2> && std::is_same_v<T1, T1_> && std::is_same_v<T2, T2_>;

template <typename Cpu, typename Mode, typename Vertex, typename Value, typename AtomicT>
class data_to_relax_base {};

template <typename Cpu, typename Vertex, typename Value, typename AtomicT>
class data_to_relax_base<Cpu, mode::distances, Vertex, Value, AtomicT> {
public:
    using value_type = Value;
    using atomic_type = AtomicT;
    using atomic_value_type = typename AtomicT::value_type;
    using atomic_value_allocator_type = inner_alloc<AtomicT>;
    using vertex_type = Vertex;

    data_to_relax_base(const std::int64_t& vertex_count,
                       const Vertex& source,
                       const value_type& max_dist,
                       byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              max_dist(max_dist),
              atomic_value_allocator(alloc_ptr) {
        distances = allocate(atomic_value_allocator, vertex_count);
        distances = new (distances) atomic_type[vertex_count]();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            store(i, max_dist);
        });
        store(source, 0);
    }

    virtual ~data_to_relax_base() {
        deallocate(atomic_value_allocator, distances, vertex_count);
    }

    inline atomic_type* get_distances_ptr() {
        return distances;
    }

    template <
        typename V = value_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_not_class_params_the_same_v<V, AV, value_type, atomic_value_type> &&
                             std::is_same_v<value_type, double> &&
                             std::is_same_v<atomic_value_type, std::int64_t>,
                         bool> = true>
    inline double load(const Vertex& u) {
        std::int64_t a_int = distances[u].load();
        std::int64_t* a_int_ptr = &a_int;
        return *reinterpret_cast<double*>(a_int_ptr);
    }

    template <typename V = value_type,
              typename AV = atomic_value_type,
              std::enable_if_t<is_class_params_the_same_v<V, AV, value_type, atomic_value_type>,
                               bool> = true>
    inline void store(const Vertex& u, const value_type& value) {
        distances[u].store(value);
    }

    template <typename V = value_type,
              typename AV = atomic_value_type,
              std::enable_if_t<is_class_params_the_same_v<V, AV, value_type, atomic_value_type>,
                               bool> = true>
    inline bool compare_exchange_strong(const Vertex& u,
                                        value_type& old_value,
                                        value_type new_value) {
        return distances[u].compare_exchange_strong(old_value, new_value);
    }

    template <typename V = value_type,
              typename AV = atomic_value_type,
              std::enable_if_t<is_class_params_the_same_v<V, AV, value_type, atomic_value_type>,
                               bool> = true>
    inline value_type load(const Vertex& u) {
        return distances[u].load();
    }

    template <
        typename V = value_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_not_class_params_the_same_v<V, AV, value_type, atomic_value_type> &&
                             std::is_same_v<value_type, double> &&
                             std::is_same_v<atomic_value_type, std::int64_t>,
                         bool> = true>
    inline void store(const Vertex& u, const double& value) {
        distances[u].store(*reinterpret_cast<const std::int64_t*>(&value));
    }

    template <
        typename V = value_type,
        typename AV = atomic_value_type,
        std::enable_if_t<is_not_class_params_the_same_v<V, AV, value_type, atomic_value_type> &&
                             std::is_same_v<value_type, double> &&
                             std::is_same_v<atomic_value_type, std::int64_t>,
                         bool> = true>
    inline bool compare_exchange_strong(const Vertex& u, double& old_value, double new_value) {
        double* old_value_ptr = &old_value;
        double* new_value_ptr = &new_value;
        std::int64_t old_value_int_representation = *reinterpret_cast<std::int64_t*>(old_value_ptr);
        std::int64_t new_value_int_representation = *reinterpret_cast<std::int64_t*>(new_value_ptr);
        return distances[u].compare_exchange_strong(old_value_int_representation,
                                                    new_value_int_representation);
    }

private:
    const std::int64_t vertex_count;
    const value_type max_dist;
    atomic_value_allocator_type atomic_value_allocator;
    atomic_type* distances;
};

template <typename Cpu, typename Mode, typename Vertex, typename Value>
class data_to_relax : public data_to_relax_base<Cpu, Mode, Vertex, Value, std::atomic<Value>> {};

template <typename Cpu, typename Vertex, typename Value>
class data_to_relax<Cpu, mode::distances, Vertex, Value>
        : public data_to_relax_base<Cpu, mode::distances, Vertex, Value, std::atomic<Value>> {
    using data_to_relax_base<Cpu, mode::distances, Vertex, Value, std::atomic<Value>>::
        data_to_relax_base;
};

template <typename Cpu, typename Vertex>
class data_to_relax<Cpu, mode::distances, Vertex, double>
        : public data_to_relax_base<Cpu,
                                    mode::distances,
                                    Vertex,
                                    double,
                                    std::atomic<std::int64_t>> {
    using data_to_relax_base<Cpu, mode::distances, Vertex, double, std::atomic<std::int64_t>>::
        data_to_relax_base;
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

template <typename Topology, typename EdgeValue, typename BinsVector, typename DataToRelax>
inline void relax_edges(const Topology& t,
                        const EdgeValue* vals,
                        typename Topology::vertex_type u,
                        EdgeValue delta,
                        DataToRelax& dist,
                        BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        EdgeValue old_dist = dist.load(v);
        EdgeValue new_dist = dist.load(u) + v_w;
        while (new_dist < old_dist) {
            if (dist.compare_exchange_strong(v, old_dist, new_dist)) {
                update_bins(v, new_dist, delta, local_bins);
                break;
            }
            old_dist = dist.load(v);
        }
    }
}

template <typename BinsVector>
inline void find_next_bin_index_seq(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    std::int64_t min_nonempty_bucket_index = max_bin_count;
    for (std::int64_t i = curr_bin_index; i < local_bins[0].size(); i++) {
        if (!local_bins[0][i].empty()) {
            min_nonempty_bucket_index = std::min(max_bin_count, i);
            break;
        }
    }
    curr_bin_index = min_nonempty_bucket_index;
}

template <typename BinsVector>
inline void find_next_bin_index_thr(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    curr_bin_index = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        (std::int64_t)local_bins.size(),
        (std::int64_t)max_bin_count,
        [&](std::int64_t begin, std::int64_t end, std::int64_t thread_min_index) -> std::int64_t {
            for (std::int64_t id = begin; id < end; ++id) {
                for (std::int64_t i = curr_bin_index; i < local_bins[id].size(); i++) {
                    if (!local_bins[id][i].empty()) {
                        thread_min_index = std::min(max_bin_count, i);
                        break;
                    }
                }
            }
            return thread_min_index;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return std::min(x, y);
        });
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin_seq(const std::int64_t& curr_bin_index,
                                             BinsVector& local_bins,
                                             SharedBinContainer& shared_bin) {
    std::int64_t vertex_count_in_shared_bin = 0;
    if (curr_bin_index < local_bins[0].size()) {
        if (local_bins[0][curr_bin_index].size() != 0) {
            vertex_count_in_shared_bin = local_bins[0][curr_bin_index].size();
            copy(local_bins[0][curr_bin_index].begin(),
                 local_bins[0][curr_bin_index].end(),
                 shared_bin.get_mutable_data());
            local_bins[0][curr_bin_index].resize(0);
        }
    }
    return vertex_count_in_shared_bin;
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin_thr(const std::int64_t& curr_bin_index,
                                             BinsVector& local_bins,
                                             SharedBinContainer& shared_bin) {
    std::atomic<std::int64_t> vertex_count_in_shared_bin = 0;
    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        int thread_id = dal::detail::threader_get_current_thread_index();
        if (curr_bin_index < local_bins[thread_id].size()) {
            if (local_bins[thread_id][curr_bin_index].size() != 0) {
                std::int64_t copy_start = vertex_count_in_shared_bin.fetch_add(
                    local_bins[thread_id][curr_bin_index].size());
                copy(local_bins[thread_id][curr_bin_index].begin(),
                     local_bins[thread_id][curr_bin_index].end(),
                     shared_bin.get_mutable_data() + copy_start);
                local_bins[thread_id][curr_bin_index].resize(0);
            }
        }
    });
    return vertex_count_in_shared_bin.load();
}

template <typename Cpu, typename EdgeValue>
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

        data_to_relax<Cpu, mode::distances, vertex_type, value_type> dist(vertex_count,
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
                    const vertex_type& u = shared_bin[i];
                    if (dist.load(u) >= delta * static_cast<value_type>(curr_bin_index)) {
                        relax_edges(t,
                                    vals,
                                    u,
                                    delta,
                                    dist,
                                    local_bins[dal::detail::threader_get_current_thread_index()]);
                    }
                });

            dal::detail::threader_for(thread_cnt, thread_cnt, [&](std::int64_t i) {
                const std::int64_t thread_id = dal::detail::threader_get_current_thread_index();
                auto& local_bin = local_bins[thread_id];
                while (curr_bin_index < local_bin.size() && !local_bin[curr_bin_index].empty() &&
                       local_bin[curr_bin_index].size() < max_elements_in_bin) {
                    auto copy_begin = local_processing_bins + max_elements_in_bin * thread_id;
                    copy(local_bin[curr_bin_index].begin(),
                         local_bin[curr_bin_index].end(),
                         copy_begin);
                    const std::int64_t copy_count = local_bin[curr_bin_index].size();
                    local_bin[curr_bin_index].resize(0);
                    for (std::int64_t j = 0; j < copy_count; ++j) {
                        relax_edges(t, vals, copy_begin[j], delta, dist, local_bin);
                    }
                }
            });

            find_next_bin_index_thr(curr_bin_index, local_bins);

            vertex_count_in_shared_bin =
                reduce_to_common_bin_thr(curr_bin_index, local_bins, shared_bin);
        }

        deallocate(vertex_allocator, local_processing_bins, max_elements_in_bin * thread_cnt);

        auto dist_arr = array<value_type>::empty(vertex_count);
        value_type* dist_ = dist_arr.get_mutable_data();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist_[i] = dist.load(i);
        });

        return traverse_result<task::one_to_all>().set_distances(
            dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
    }
};

template <typename EV, typename VT>
struct dist_pred {
    dist_pred(const EV& dist_, const VT& pred_) : dist(dist_), pred(pred_) {}
    EV dist = 0;
    VT pred = 0;
};

template <class T1, class T2>
bool operator==(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return lhs.dist == rhs.dist && lhs.pred == rhs.pred;
}

template <class T1, class T2>
bool operator!=(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return !(lhs == rhs);
}

template <typename Topology, typename EdgeValue, typename DP, typename BinsVector>
inline void relax_edges_with_pred_seq(const Topology& t,
                                      const EdgeValue* vals,
                                      typename Topology::vertex_type u,
                                      EdgeValue delta,
                                      DP* dp,
                                      BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        auto old_dp = dp[v];
        const EdgeValue new_dist = dp[u].dist + v_w;
        if (new_dist < old_dp.dist) {
            dp[v] = dist_pred(new_dist, u);
            update_bins(v, new_dist, delta, local_bins);
        }
    }
}

template <typename Cpu, typename EdgeValue>
struct delta_stepping_with_pred_sequential {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        using value_type = EdgeValue;
        using vertex_type = std::int32_t;
        using vp_type = dist_pred<value_type, vertex_type>;
        using vp_allocator_type = inner_alloc<vp_type>;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);
        vp_allocator_type vp_allocator(alloc_ptr);

        const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());

        const value_type delta = desc.get_delta();

        const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t max_elements_in_bin = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();

        vp_type* dp = allocate(vp_allocator, vertex_count);
        for (std::int64_t i = 0; i < vertex_count; ++i) {
            new (dp + i) dist_pred<value_type, vertex_type>(max_dist, -1);
        }

        dp[source] = dist_pred<value_type, vertex_type>(0, -1);

        vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                        vertex_allocator);

        shared_bin[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t vertex_count_in_shared_bin = 1;

        using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
        using v1a_t = inner_alloc<v1v_t>;

        using v2v_t = vector_container<v1v_t, v1a_t>;
        using v2a_t = inner_alloc<v2v_t>;
        using v3v_t = vector_container<v2v_t, v2a_t>;

        v2a_t v2a(alloc_ptr);
        v3v_t local_bins(1, v2a);

        while (curr_bin_index != max_bin_count) {
            for (std::int64_t i = 0; i < vertex_count_in_shared_bin; ++i) {
                vertex_type u = shared_bin[i];
                if (dp[u].dist >= delta * static_cast<value_type>(curr_bin_index)) {
                    relax_edges_with_pred_seq(t, vals, u, delta, dp, local_bins[0]);
                }
            }

            while (curr_bin_index < local_bins[0].size() &&
                   !local_bins[0][curr_bin_index].empty() &&
                   local_bins[0][curr_bin_index].size() < max_elements_in_bin) {
                vector_container<vertex_type> curr_bin_copy(local_bins[0][curr_bin_index].size());
                copy(local_bins[0][curr_bin_index].begin(),
                     local_bins[0][curr_bin_index].end(),
                     curr_bin_copy.begin());

                local_bins[0][curr_bin_index].resize(0);
                for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j) {
                    relax_edges_with_pred_seq(t, vals, curr_bin_copy[j], delta, dp, local_bins[0]);
                }
            }

            find_next_bin_index_seq(curr_bin_index, local_bins);

            vertex_count_in_shared_bin =
                reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);
        }

        if (desc.get_optional_results() & optional_results::distances) {
            auto dist_arr = array<value_type>::empty(vertex_count);
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();
            vertex_type* pred_ = pred_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                const auto dp_i = dp[i];
                dist_[i] = dp_i.dist;
                pred_[i] = dp_i.pred;
            }

            deallocate(vp_allocator, dp, vertex_count);
            return traverse_result<task::one_to_all>()
                .set_distances(dal::detail::homogen_table_builder{}
                                   .reset(dist_arr, t.get_vertex_count(), 1)
                                   .build())
                .set_predecessors(dal::detail::homogen_table_builder{}
                                      .reset(pred_arr, t.get_vertex_count(), 1)
                                      .build());
        }
        else {
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* pred_ = pred_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                const auto dp_i = dp[i];
                pred_[i] = dp_i.pred;
            }

            deallocate(vp_allocator, dp, vertex_count);
            return traverse_result<task::one_to_all>().set_predecessors(
                dal::detail::homogen_table_builder{}
                    .reset(pred_arr, t.get_vertex_count(), 1)
                    .build());
        }
    }
};

template <typename Cpu, typename EdgeValue>
struct delta_stepping_with_pred {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        return delta_stepping_with_pred_sequential<Cpu, EdgeValue>()(desc, t, vals, alloc_ptr);
    }
};

} // namespace oneapi::dal::preview::shortest_paths::backend

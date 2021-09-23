
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
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/detail/threading.hpp"

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

} // namespace oneapi::dal::preview::shortest_paths::backend

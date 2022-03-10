/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Float>
array<std::int64_t>
triangle_counting<Float, task::local, dal::preview::detail::topology<std::int32_t>, automatic>::
operator()(const dal::detail::host_policy& policy,
           const dal::preview::detail::topology<std::int32_t>& t,
           std::int64_t* triangles_local) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::triangle_counting_local<decltype(cpu)>(t, triangles_local);
    });
}

template <typename Float>
std::int64_t
triangle_counting<Float, task::global, dal::preview::detail::topology<std::int32_t>, scalar>::
operator()(const dal::detail::host_policy& policy,
           const dal::preview::detail::topology<std::int32_t>& t) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::triangle_counting_global_scalar<decltype(cpu)>(t);
    });
}

template <typename Float>
std::int64_t
triangle_counting<Float, task::global, dal::preview::detail::topology<std::int32_t>, vector>::
operator()(const dal::detail::host_policy& policy,
           const dal::preview::detail::topology<std::int32_t>& t) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::triangle_counting_global_vector<decltype(cpu)>(t);
    });
}

template <typename Float>
std::int64_t triangle_counting<Float,
                               task::global,
                               dal::preview::detail::topology<std::int32_t>,
                               vector,
                               relabeled>::operator()(const dal::detail::host_policy& policy,
                                                      const std::int32_t* vertex_neighbors,
                                                      const std::int64_t* edge_offsets,
                                                      const std::int32_t* degrees,
                                                      std::int64_t vertex_count,
                                                      std::int64_t edge_count) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::triangle_counting_global_vector_relabel<decltype(cpu)>(vertex_neighbors,
                                                                               edge_offsets,
                                                                               degrees,
                                                                               vertex_count,
                                                                               edge_count);
    });
}

std::int64_t compute_global_triangles(const dal::detail::host_policy& policy,
                                      const array<std::int64_t>& local_triangles,
                                      std::int64_t vertex_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::compute_global_triangles<decltype(cpu)>(local_triangles, vertex_count);
    });
}

template struct ONEDAL_EXPORT
    triangle_counting<float, task::local, dal::preview::detail::topology<std::int32_t>, automatic>;

template struct ONEDAL_EXPORT
    triangle_counting<float, task::global, dal::preview::detail::topology<std::int32_t>, scalar>;

template struct ONEDAL_EXPORT
    triangle_counting<float, task::global, dal::preview::detail::topology<std::int32_t>, vector>;

template struct ONEDAL_EXPORT triangle_counting<float,
                                                task::global,
                                                dal::preview::detail::topology<std::int32_t>,
                                                vector,
                                                relabeled>;

} // namespace oneapi::dal::preview::triangle_counting::detail

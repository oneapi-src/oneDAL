/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

std::int64_t triangle_counting_global_scalar(const dal::detail::host_policy& policy,
                                             const std::int32_t* vertex_neighbors,
                                             const std::int64_t* edge_offsets,
                                             const std::int32_t* degrees,
                                             std::int64_t vertex_count,
                                             std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_scalar_cpu<decltype(cpu)>(vertex_neighbors,
                                                                  edge_offsets,
                                                                  degrees,
                                                                  vertex_count,
                                                                  edge_count);
    });
}

std::int64_t triangle_counting_global_vector(const dal::detail::host_policy& policy,
                                             const std::int32_t* vertex_neighbors,
                                             const std::int64_t* edge_offsets,
                                             const std::int32_t* degrees,
                                             std::int64_t vertex_count,
                                             std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_vector_cpu<decltype(cpu)>(vertex_neighbors,
                                                                  edge_offsets,
                                                                  degrees,
                                                                  vertex_count,
                                                                  edge_count);
    });
}

std::int64_t triangle_counting_global_vector_relabel(const dal::detail::host_policy& policy,
                                                     const std::int32_t* vertex_neighbors,
                                                     const std::int64_t* edge_offsets,
                                                     const std::int32_t* degrees,
                                                     std::int64_t vertex_count,
                                                     std::int64_t edge_count) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return triangle_counting_global_vector_relabel_cpu<decltype(cpu)>(vertex_neighbors,
                                                                          edge_offsets,
                                                                          degrees,
                                                                          vertex_count,
                                                                          edge_count);
    });
}

/*
template <typename Task, typename Float, typename Method>
vertex_ranking_result<Task> backend_default<dal::detail::host_policy, Task, 
                                         Float,
                                         Method,
                                         dal::preview::detail::topology<std::int32_t>>::
operator()(const dal::detail::host_policy &policy,
           const descriptor_base<Task> &desc,
           const dal::preview::detail::topology<std::int32_t> &data) {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return call_triangle_counting_default_kernel_int32<decltype(cpu)>(desc, data);
    });
}

s


template struct ONEDAL_EXPORT backend_default<dal::detail::host_policy, dal::preview::triangle_counting::task::local, 
                                              float,
                                              dal::preview::triangle_counting::method::ordered_count,
                                              dal::preview::detail::topology<std::int32_t>>;

template struct ONEDAL_EXPORT backend_default<dal::detail::host_policy, dal::preview::triangle_counting::task::global, 
                                              float,
                                              dal::preview::triangle_counting::method::ordered_count,
                                              dal::preview::detail::topology<std::int32_t>>;
*/
} // namespace oneapi::dal::preview::triangle_counting::detail

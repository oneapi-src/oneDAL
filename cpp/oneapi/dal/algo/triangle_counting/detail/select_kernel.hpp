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

#pragma once

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Policy, typename Task, typename Topology>
struct ONEDAL_EXPORT backend_base {
    virtual vertex_ranking_result<Task> operator()(const Policy &ctx,
                                                const descriptor_base<Task> &descriptor,
                                                const Topology &data) = 0;
    virtual ~backend_base() {}
};

template <typename Policy, typename Task, typename Float, typename Method, typename Topology>
struct ONEDAL_EXPORT backend_default : public backend_base<Policy, Task, Topology> {
    virtual vertex_ranking_result<Task> operator()(const Policy &ctx,
                                                const descriptor_base<Task> &descriptor,
                                                const Topology &data) {
        return call_triangle_counting_default_kernel_general(descriptor, data);
    }
    virtual ~backend_default() {}
};

template <typename Task, typename Float, typename Method>
struct backend_default<dal::detail::host_policy, Task, 
                       Float,
                       Method,
                       dal::preview::detail::topology<std::int32_t>>
        : public backend_base<dal::detail::host_policy, Task, 
                              dal::preview::detail::topology<std::int32_t>> {
    virtual vertex_ranking_result<Task> operator()(
        const dal::detail::host_policy &ctx,
        const descriptor_base<Task> &descriptor,
        const dal::preview::detail::topology<std::int32_t> &data);
    virtual ~backend_default() {}
};

template <typename Policy, typename Task, typename Float, class Method, typename Topology>
dal::detail::pimpl<backend_base<Policy, Task, Topology>> get_backend(const descriptor_base<Task> &desc,
                                                               const Topology &data) {
    return dal::detail::pimpl<backend_base<Policy, Task, Topology>>(
        new backend_default<Policy, Task, float, method::by_default, Topology>);
}

} // namespace oneapi::dal::preview::triangle_counting::detail

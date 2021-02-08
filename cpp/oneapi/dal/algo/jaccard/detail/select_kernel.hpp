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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/detail/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Policy, typename Topology>
struct ONEDAL_EXPORT backend_base {
    virtual vertex_similarity_result operator()(const Policy &ctx,
                                                const descriptor_base &descriptor,
                                                const Topology &data,
                                                void *result_ptr) = 0;
    virtual ~backend_base() {}
};

template <typename Policy, typename Float, typename Method, typename Topology>
struct ONEDAL_EXPORT backend_default : public backend_base<Policy, Topology> {
    virtual vertex_similarity_result operator()(const Policy &ctx,
                                                const descriptor_base &descriptor,
                                                const Topology &data,
                                                void *result_ptr) {
        return call_jaccard_default_kernel_general(descriptor, data, result_ptr);
    }
    virtual ~backend_default() {}
};

template <typename Float, typename Method>
struct backend_default<dal::detail::host_policy,
                       Float,
                       Method,
                       dal::preview::detail::topology<std::int32_t>>
        : public backend_base<dal::detail::host_policy,
                              dal::preview::detail::topology<std::int32_t>> {
    virtual vertex_similarity_result operator()(
        const dal::detail::host_policy &ctx,
        const descriptor_base &descriptor,
        const dal::preview::detail::topology<std::int32_t> &data,
        void *result_ptr);
    virtual ~backend_default() {}
};

template <typename Policy, typename Float, class Method, typename Topology>
dal::detail::pimpl<backend_base<Policy, Topology>> get_backend(const descriptor_base &desc,
                                                               const Topology &data) {
    return dal::detail::pimpl<backend_base<Policy, Topology>>(
        new backend_default<Policy, float, method::by_default, Topology>);
}

} // namespace oneapi::dal::preview::jaccard::detail

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

#include "oneapi/dal/algo/louvain/detail/vertex_partitioning_default_kernel.hpp"
#include "oneapi/dal/algo/louvain/backend/cpu/vertex_partitioning_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::louvain::detail {

template <typename Float, typename EdgeValue>
vertex_partitioning_result<task::vertex_partitioning> louvain_kernel<
    Float,
    task::vertex_partitioning,
    dal::preview::detail::topology<std::int32_t>,
    EdgeValue>::operator()(const dal::detail::host_policy &policy,
                           const detail::descriptor_base<task::vertex_partitioning> &desc,
                           const dal::preview::detail::topology<std::int32_t> &t,
                           const std::int32_t *init_partition,
                           const EdgeValue *vals,
                           byte_alloc_iface *alloc_ptr) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::louvain_kernel<decltype(cpu), Float, EdgeValue>{}(desc,
                                                                          t,
                                                                          init_partition,
                                                                          vals,
                                                                          alloc_ptr);
    });
}

template struct ONEDAL_EXPORT louvain_kernel<float,
                                             task::vertex_partitioning,
                                             dal::preview::detail::topology<std::int32_t>,
                                             std::int32_t>;

template struct ONEDAL_EXPORT louvain_kernel<float,
                                             task::vertex_partitioning,
                                             dal::preview::detail::topology<std::int32_t>,
                                             double>;

template struct ONEDAL_EXPORT louvain_kernel<double,
                                             task::vertex_partitioning,
                                             dal::preview::detail::topology<std::int32_t>,
                                             std::int32_t>;

template struct ONEDAL_EXPORT louvain_kernel<double,
                                             task::vertex_partitioning,
                                             dal::preview::detail::topology<std::int32_t>,
                                             double>;

} // namespace oneapi::dal::preview::louvain::detail

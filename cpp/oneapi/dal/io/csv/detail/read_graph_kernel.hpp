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

#pragma once

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/io/csv/detail/read_graph_kernel_impl.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::csv::detail {

template <typename Descriptor>
inline typename Descriptor::object_t read_graph_default_kernel(
    const dal::detail::host_policy& ctx,
    const dal::csv::detail::data_source_base& ds,
    const Descriptor& desc) {
    using graph_t = typename Descriptor::object_t;
    using weight_t = typename dal::preview::edge_user_value_type<graph_t>;
    constexpr bool is_edge_weighted =
        !oneapi::dal::detail::is_one_of_v<weight_t, oneapi::dal::preview::empty_value>;
    using read_impl_t = read_impl<is_edge_weighted, Descriptor, dal::csv::detail::data_source_base>;
    graph_t graph;
    read_impl_t{}(ds, desc, graph);
    return graph;
}
} // namespace oneapi::dal::preview::csv::detail

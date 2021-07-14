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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/io/csv/detail/read_graph_kernel_impl.hpp"

namespace oneapi::dal::csv::detail {

template <typename Descriptor>
inline typename Descriptor::object_t read_graph_default_kernel(const dal::detail::host_policy& ctx,
                                                               const detail::data_source_base& ds,
                                                               const Descriptor& desc) {
    using object_t = typename Descriptor::object_t;
    object_t graph;
    oneapi::dal::preview::read_graph::detail::read_impl(ds, desc, graph);
    return graph;
}
} // namespace oneapi::dal::csv::detail

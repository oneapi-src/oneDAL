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

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/detail/vertex_partitioning_ops.hpp"
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/vertex_partitioning.hpp"

namespace oneapi::dal::preview::detail {

template <typename Descriptor, typename Graph>
struct vertex_partitioning_ops<Descriptor, Graph, louvain::detail::descriptor_tag>
        : louvain::detail::vertex_partitioning_ops<Descriptor, Graph> {};

} // namespace oneapi::dal::preview::detail

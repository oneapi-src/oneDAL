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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_ops.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/vertex_ranking.hpp"

namespace oneapi::dal::preview::detail {

template <typename Descriptor, typename Graph>
struct vertex_ranking_ops<Descriptor, Graph, triangle_counting::detail::descriptor_tag>
        : triangle_counting::detail::vertex_ranking_ops<Descriptor, Graph> {};

} // namespace oneapi::dal::preview::detail

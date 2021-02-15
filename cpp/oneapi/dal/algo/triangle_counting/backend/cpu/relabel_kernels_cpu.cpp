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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/relabel_kernels.hpp"
#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview {
namespace triangle_counting {
namespace detail {

template void sort_ids_by_degree_cpu<__CPU_TAG__>(
    const std::int32_t* degrees,
    std::pair<std::int32_t, std::size_t>* degree_id_pairs,
    std::int64_t vertex_count);

template void fill_new_degrees_and_ids_cpu<__CPU_TAG__>(
    std::pair<std::int32_t, std::size_t>* degree_id_pairs,
    std::int32_t* new_ids,
    std::int32_t* degrees_relabel,
    std::int64_t vertex_count);

template void parallel_prefix_sum_cpu<__CPU_TAG__>(std::int32_t* degrees_relabel,
                                                   std::int64_t* offsets,
                                                   std::int64_t* part_prefix,
                                                   std::int64_t* local_sums,
                                                   size_t block_size,
                                                   std::int64_t num_blocks,
                                                   std::int64_t vertex_count);

template void fill_relabeled_topology_cpu<__CPU_TAG__>(const std::int32_t* vertex_neighbors,
                                                       const std::int64_t* edge_offsets,
                                                       std::int32_t* vertex_neighbors_relabel,
                                                       std::int64_t* edge_offsets_relabel,
                                                       std::int64_t* offsets,
                                                       std::int32_t* new_ids,
                                                       std::int64_t vertex_count);

} // namespace detail
} // namespace triangle_counting
} // namespace oneapi::dal::preview

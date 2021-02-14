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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel_scalar.hpp"
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
/*
template vertex_ranking_result<task::local> call_triangle_counting_default_kernel_scalar<__CPU_TAG__, std::int32_t>(
    const detail::descriptor_base<task::local> &desc,
    const dal::preview::detail::topology<std::int32_t> &data);

template vertex_ranking_result<task::local> call_triangle_counting_default_kernel_scalar<__CPU_TAG__, std::int64_t>(
    const detail::descriptor_base<task::local> &desc,
    const dal::preview::detail::topology<std::int64_t> &data);

template <>
vertex_ranking_result<task::local> call_triangle_counting_default_kernel_int32<__CPU_TAG__>(
    const detail::descriptor_base<task::local> &desc,
    const dal::preview::detail::topology<std::int32_t> &data) {
    return call_triangle_counting_default_kernel_scalar<__CPU_TAG__>(desc, data);
}


template vertex_ranking_result<task::global> call_triangle_counting_default_kernel_scalar<__CPU_TAG__, std::int32_t>(
    const detail::descriptor_base<task::global> &desc,
    const dal::preview::detail::topology<std::int32_t> &data);

template vertex_ranking_result<task::global> call_triangle_counting_default_kernel_scalar<__CPU_TAG__, std::int64_t>(
    const detail::descriptor_base<task::global> &desc,
    const dal::preview::detail::topology<std::int64_t> &data);

template <>
vertex_ranking_result<task::global> call_triangle_counting_default_kernel_int32<__CPU_TAG__>(
    const detail::descriptor_base<task::global> &desc,
    const dal::preview::detail::topology<std::int32_t> &data) {
    return call_triangle_counting_default_kernel_scalar<__CPU_TAG__>(desc, data);
}*/

template std::int64_t triangle_counting_global_scalar_novec<__CPU_TAG__>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count);

template std::int64_t triangle_counting_global_vector_novec<__CPU_TAG__>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count);

template std::int64_t triangle_counting_global_vector_relabel_novec<__CPU_TAG__>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count);

template <>
std::int64_t triangle_counting_global_scalar_cpu<__CPU_TAG__>(const std::int32_t* vertex_neighbors,
                                                              const std::int64_t* edge_offsets,
                                                              const std::int32_t* degrees,
                                                              std::int64_t vertex_count,
                                                              std::int64_t edge_count) {
    return triangle_counting_global_scalar_novec<__CPU_TAG__>(vertex_neighbors,
                                                              edge_offsets,
                                                              degrees,
                                                              vertex_count,
                                                              edge_count);
}

template <>
std::int64_t triangle_counting_global_vector_cpu<__CPU_TAG__>(const std::int32_t* vertex_neighbors,
                                                              const std::int64_t* edge_offsets,
                                                              const std::int32_t* degrees,
                                                              std::int64_t vertex_count,
                                                              std::int64_t edge_count) {
    return triangle_counting_global_vector_novec<__CPU_TAG__>(vertex_neighbors,
                                                              edge_offsets,
                                                              degrees,
                                                              vertex_count,
                                                              edge_count);
}

template <>
std::int64_t triangle_counting_global_vector_relabel_cpu<__CPU_TAG__>(
    const std::int32_t* vertex_neighbors,
    const std::int64_t* edge_offsets,
    const std::int32_t* degrees,
    std::int64_t vertex_count,
    std::int64_t edge_count) {
    return triangle_counting_global_vector_relabel_novec<__CPU_TAG__>(vertex_neighbors,
                                                                      edge_offsets,
                                                                      degrees,
                                                                      vertex_count,
                                                                      edge_count);
}

} // namespace detail
} // namespace triangle_counting
} // namespace oneapi::dal::preview

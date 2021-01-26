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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/algo/jaccard/detail/vertex_similarity_ops.hpp"

#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Policy, typename Float, class Method, typename Graph>
struct ONEDAL_EXPORT vertex_similarity_ops_dispatcher {
    vertex_similarity_result operator()(const Policy &policy,
                                        const descriptor_base &descriptor,
                                        vertex_similarity_input<Graph> &input) const;
};

template <typename Descriptor, typename Graph>
struct vertex_similarity_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = vertex_similarity_input<Graph>;
    using result_t = vertex_similarity_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor &param, vertex_similarity_input<Graph> &input) const {
        using msg = dal::detail::error_messages;

        const auto row_begin = param.get_row_range_begin();
        const auto row_end = param.get_row_range_end();
        const auto column_begin = param.get_column_range_begin();
        const auto column_end = param.get_column_range_end();
        if (row_begin < 0 || column_begin < 0) {
            throw invalid_argument(msg::negative_interval());
        }
        if (row_begin > row_end) {
            throw invalid_argument(msg::row_begin_gt_row_end());
        }
        if (column_begin > column_end) {
            throw invalid_argument(msg::column_begin_gt_column_end());
        }
        const auto vertex_count =
            dal::detail::get_impl(input.get_graph()).get_topology()._vertex_count;
        // Safe conversion as ranges were checked
        if (static_cast<std::size_t>(row_end) > vertex_count ||
            static_cast<std::size_t>(column_end) > vertex_count) {
            throw out_of_range(msg::interval_gt_vertex_count());
        }
        if (row_end >= dal::detail::limits<std::int32_t>::max() ||
            column_end >= dal::detail::limits<std::int32_t>::max()) {
            throw invalid_argument(msg::range_idx_gt_max_int32());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy,
                    const Descriptor &desc,
                    vertex_similarity_input<Graph> &input) const {
        check_preconditions(desc, input);
        return vertex_similarity_ops_dispatcher<Policy, float_t, method_t, Graph>()(policy,
                                                                                    desc,
                                                                                    input);
    }
};

template <typename Policy, typename Topology>
struct backend_base {
    virtual vertex_similarity_result operator()(const Policy &ctx,
                                                const descriptor_base &descriptor,
                                                const Topology &data,
                                                void *result_ptr) {
        return vertex_similarity_result();
    }
    virtual ~backend_base() {}
};

template <typename Index>
vertex_similarity_result call_jaccard_default_kernel_general(
    const descriptor_base &desc,
    const dal::preview::detail::topology<Index> &data,
    void *result_ptr);

template <typename Policy, typename Float, typename Method, typename Topology>
struct backend_default : public backend_base<Policy, Topology> {
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

inline std::size_t get_number_elements_in_block(const std::int32_t &row_range_begin,
                                                const std::int32_t &row_range_end,
                                                const std::int32_t &column_range_begin,
                                                const std::int32_t &column_range_end) {
    ONEDAL_ASSERT(row_range_end >= row_range_begin, "Negative interval found");
    const std::size_t row_count = row_range_end - row_range_begin;
    ONEDAL_ASSERT(column_range_end >= column_range_begin, "Negative interval found");
    const std::size_t column_count = column_range_end - column_range_begin;
    // compute the number of the vertex pairs in the block of the graph
    const std::size_t vertex_pairs_count = row_count * column_count;
    ONEDAL_ASSERT(vertex_pairs_count / row_count == column_count,
                  "Overflow found in multiplication of two values");
    return vertex_pairs_count;
}

template <typename Float, typename Index>
inline std::size_t get_max_block_size(const std::int64_t &vertex_pairs_count) {
    const std::size_t vertex_pair_element_count = 2; // 2 elements in the vertex pair
    const std::size_t jaccard_coeff_element_count = 1; // 1 Jaccard coeff for the vertex pair

    const std::size_t vertex_pair_size = vertex_pair_element_count * sizeof(Index); // size in bytes
    const std::size_t jaccard_coeff_size =
        jaccard_coeff_element_count * sizeof(Float); // size in bytes
    const std::size_t element_result_size = vertex_pair_size + jaccard_coeff_size;

    const std::size_t block_result_size = element_result_size * vertex_pairs_count;
    ONEDAL_ASSERT(block_result_size / vertex_pairs_count == element_result_size,
                  "Overflow found in multiplication of two values");
    return block_result_size;
}

template <typename Policy, typename Float, class Method, typename Graph>
vertex_similarity_result vertex_similarity_ops_dispatcher<Policy, Float, Method, Graph>::operator()(
    const Policy &policy,
    const descriptor_base &desc,
    vertex_similarity_input<Graph> &input) const {
    const auto &csr_topology =
        dal::preview::detail::csr_topology_builder<Graph>()(input.get_graph());
    const auto row_begin =
        dal::detail::integral_cast<vertex_type<Graph>>(desc.get_row_range_begin());
    const auto row_end = dal::detail::integral_cast<vertex_type<Graph>>(desc.get_row_range_end());
    const auto column_begin =
        dal::detail::integral_cast<vertex_type<Graph>>(desc.get_column_range_begin());
    const auto column_end =
        dal::detail::integral_cast<vertex_type<Graph>>(desc.get_column_range_end());
    const auto number_elements_in_block =
        get_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    const auto max_block_size =
        get_max_block_size<Float, vertex_type<Graph>>(number_elements_in_block);
    void *result_ptr = input.get_caching_builder()(max_block_size);

    static auto impl = get_backend<Policy, Float, Method>(desc, csr_topology);
    return (*impl)(policy, desc, csr_topology, result_ptr); // dep on backend
}

template <typename Index>
inline Index min(const Index &a, const Index &b) {
    return (a >= b) ? b : a;
}

template <typename Index>
inline Index max(const Index &a, const Index &b) {
    return (a <= b) ? b : a;
}

template <typename Index>
inline std::size_t intersection(const Index *neigh_u, const Index *neigh_v, Index n_u, Index n_v);

template <typename Index>
vertex_similarity_result call_jaccard_default_kernel_general(
    const descriptor_base &desc,
    const dal::preview::detail::topology<Index> &data,
    void *result_ptr) {
    const auto g_edge_offsets = data._rows.get_data();
    const auto g_vertex_neighbors = data._cols.get_data();
    const auto g_degrees = data._degrees.get_data();
    const auto row_begin = dal::detail::integral_cast<Index>(desc.get_row_range_begin());
    const auto row_end = dal::detail::integral_cast<Index>(desc.get_row_range_end());
    const auto column_begin = dal::detail::integral_cast<Index>(desc.get_column_range_begin());
    const auto column_end = dal::detail::integral_cast<Index>(desc.get_column_range_end());
    const auto number_elements_in_block =
        get_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    Index *first_vertices = reinterpret_cast<Index *>(result_ptr);
    Index *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(second_vertices + number_elements_in_block);
    std::int64_t nnz = 0;
    for (Index i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = g_degrees[i];
        const auto i_neigbhors = g_vertex_neighbors + g_edge_offsets[i];
        const auto diagonal = min(i, column_end);
        for (Index j = column_begin; j < diagonal; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz] = i;
                    second_vertices[nnz] = j;
                    // Safe incrementing of nnz
                    //max nnz = (2^(31)  * 2^(31))=(2^62) < 2^63 = max size of std::int64_t
                    nnz++;
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }
            }
        }

        if (diagonal >= column_begin && diagonal < column_end) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
        }

        for (Index j = max(column_begin, diagonal + 1); j < column_end; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz] = i;
                    second_vertices[nnz] = j;
                    nnz++;
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }
            }
        }
    }
    vertex_similarity_result res(
        homogen_table::wrap(first_vertices, number_elements_in_block, 2, data_layout::column_major),
        homogen_table::wrap(jaccard, number_elements_in_block, 1, data_layout::column_major),
        nnz);
    return res;
}

template <typename Index>
inline std::size_t intersection(const Index *neigh_u, const Index *neigh_v, Index n_u, Index n_v) {
    std::size_t total = 0;
    Index i_u = 0, i_v = 0;
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

} // namespace oneapi::dal::preview::jaccard::detail

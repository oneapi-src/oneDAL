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

/// @file
/// Contains the definition of the input and output for Jaccard Similarity
/// algorithm

#pragma once

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::preview {
namespace jaccard {

namespace detail {
template <typename Graph>
class vertex_similarity_input_impl;
class vertex_similarity_result_impl;
} // namespace detail

/// Class for the description of the input parameters of the Jaccard Similarity
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph>
class ONEAPI_DAL_EXPORT vertex_similarity_input {
public:
    /// Constructs the algorithm input initialized with the graph and the pointer
    /// to the allocated memory for the result. The size of the required memory
    /// can be computed by formula:
    /// size =  (vertex_pair_element_count * vertex_pair_size +
    ///         jaccard_coeff_element_count * jaccard_coeff_size) * vertex_pairs_count,
    /// where:
    /// - vertex_pair_element_count is 2 (vertex1, vertex2);
    /// - vertex_pair_size is (2 * sizeof(int32_t));
    /// - jaccard_coeff_element_count is 1 (coeff);
    /// - jaccard_coeff_size is (1 * sizeof(float));
    /// - vertex_pairs_count is the number of vertices in the block
    ///
    /// @param [in]   graph  The input graph
    /// @param [in/out]  result_ptr  The pointer to the allocated memory
    /// for storing the result
    vertex_similarity_input(const Graph& graph, void* result_ptr);

    /// Returns the constant reference to the input graph
    const Graph& get_graph() const;

    /// Returns the input pointer to the result
    void* get_result_ptr();

private:
    dal::detail::pimpl<detail::vertex_similarity_input_impl<Graph>> impl_;
};

/// Class for the description of the result of the Jaccard Similarity algorithm
class ONEAPI_DAL_EXPORT vertex_similarity_result {
public:
    /// Constructs the empty result
    vertex_similarity_result(){};

    /// Constructs the algorithm result initialized with the table of vertex pairs,
    /// the table of the corresponding computed Jaccard similarity coefficients and
    /// the number of non-zero Jaccard similarity coefficients in the block.
    ///
    /// @param [in]   vertex_pairs  The table of size 2*nonzero_coeff_count with
    ///                             vertex pairs which have non-zero Jaccard
    ///                             similarity coefficients
    /// @param [in]   coeffs        The table of size 1*nonzero_coeff_count with
    ///                             non-zero Jaccard similarity coefficients
    ///
    /// @param [in] nonzero_coeff_count The number of non-zero Jaccard coefficients
    vertex_similarity_result(const table& vertex_pairs,
                             const table& coeffs,
                             int64_t& nonzero_coeff_count);

    /// Returns the table of size 1*nonzero_coeff_count with non-zero Jaccard
    /// similarity coefficients
    table& get_coeffs() const;

    /// Returns the table of size 2*nonzero_coeff_count with vertex pairs which have
    /// non-zero Jaccard similarity coefficients
    table& get_vertex_pairs() const;

    /// The number of non-zero Jaccard similarity coefficients in the block
    int64_t get_nonzero_coeff_count() const;

private:
    dal::detail::pimpl<detail::vertex_similarity_result_impl> impl_;
};
} // namespace jaccard
} // namespace oneapi::dal::preview

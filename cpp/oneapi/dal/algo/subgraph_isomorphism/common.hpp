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
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {
namespace detail {
struct tag {};
class descriptor_impl;

} // namespace detail

namespace method {
struct fast {};
using by_default = fast;
} // namespace method

enum class kind { induced, non_induced };

/// The base class for the Subgraph Isomorphism algorithm descriptor
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;

    /// Constructs the empty descriptor
    descriptor_base();

    /// Returns the kind of searched subgraph which is isomorphic to pattern graph
    auto get_kind() const -> subgraph_isomorphism::kind;

    /// Returns if semantic search is required
    auto get_semantic_match() const -> bool;

    /// Returns the maximum number of matches to search
    auto get_max_match_count() const -> std::int64_t;

protected:
    void set_kind_impl(kind value);
    void set_semantic_match_impl(bool semantic_match);
    void set_max_match_count_impl(std::int64_t max_match_count);

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

/// Class for the Subgraph Isomorphism algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
template <typename Float = descriptor_base::float_t,
          typename Method = descriptor_base::method_t,
          typename Allocator = std::allocator<char>>
class descriptor : public descriptor_base {
public:
    using float_t = Float;
    using method_t = Method;
    using allocator_t = Allocator;

    explicit descriptor(Allocator allocator) {
        _alloc = allocator;
    }

    /// Sets the type of searched subgraph in Subgraph Isomorphism computation
    ///
    /// @param [in] value  The begin of the row of the graph block
    auto& set_kind(kind value) {
        this->set_kind_impl(value);
        return *this;
    }

    /// Sets the flag if semantic search is requred in Subgraph Isomorphism computation
    ///
    /// @param [in] semantic_match The flag if semantic search is requred
    auto& set_semantic_match(bool semantic_match) {
        this->set_semantic_match_impl(semantic_match);
        return *this;
    }

    /// Sets the maximum number of matchings to search in Subgraph Isomorphism computation
    ///
    /// @param [in] max_match_count  The maximum number of matchings
    auto& set_max_match_count(std::int64_t max_match_count) {
        this->set_max_match_count_impl(max_match_count);
        return *this;
    }
    Allocator get_allocator() const {
        return _alloc;
    }

private:
    Allocator _alloc;
};

namespace detail {

template <typename Graph>
constexpr bool is_valid_graph =
    dal::detail::is_one_of_v<Graph,
                             undirected_adjacency_vector_graph<vertex_user_value_type<Graph>,
                                                               edge_user_value_type<Graph>,
                                                               graph_user_value_type<Graph>,
                                                               vertex_type<Graph>,
                                                               graph_allocator<Graph>>>;

} // namespace detail

} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview

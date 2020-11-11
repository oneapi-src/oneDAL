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
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {
struct tag {};
class descriptor_impl;
} // namespace detail

namespace method {
struct fast {};
using by_default = fast;
} // namespace method

/// The base class for the Jaccard similarity algorithm descriptor
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;

    /// Constructs the empty descriptor
    descriptor_base();

    /// Returns the begin of the row of the graph block
    auto get_row_range_begin() const -> std::int64_t;

    /// Returns the end of the row of the graph block
    auto get_row_range_end() const -> std::int64_t;

    /// Returns the begin of the column of the graph block
    auto get_column_range_begin() const -> std::int64_t;

    /// Returns the end of the column of the graph block
    auto get_column_range_end() const -> std::int64_t;

protected:
    void set_row_range_impl(std::int64_t begin, std::int64_t end);
    void set_column_range_impl(std::int64_t begin, std::int64_t end);
    void set_block_impl(const std::initializer_list<std::int64_t>& row_range,
                        const std::initializer_list<std::int64_t>& column_range);

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

/// Class for the Jaccard similarity algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
template <typename Float = descriptor_base::float_t, typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using method_t = Method;

    /// Sets the range of the rows of the graph block for Jaccard similarity computation
    ///
    /// @param [in] begin  The begin of the row of the graph block
    /// @param [in] end    The end of the row of the graph block
    auto& set_row_range(std::int64_t begin, std::int64_t end) {
        this->set_row_range_impl(begin, end);
        return *this;
    }

    /// Sets the range of the columns of the graph block for Jaccard similarity computation
    ///
    /// @param [in] begin  The begin of the column of the graph block
    /// @param [in] end    The end of the column of the graph block
    auto& set_column_range(std::int64_t begin, std::int64_t end) {
        this->set_column_range_impl(begin, end);
        return *this;
    }

    /// Sets the range of the rows and columns of the graph block for Jaccard similarity
    /// computation
    ///
    /// @param [in] row_range     The range of the rows of the graph block
    /// @param [in] column_range  The range of the columns of the graph block
    auto& set_block(const std::initializer_list<std::int64_t>& row_range,
                    const std::initializer_list<std::int64_t>& column_range) {
        this->set_block_impl(row_range, column_range);
        return *this;
    }
};

/// Structure for the caching builder
struct ONEDAL_EXPORT caching_builder {
    /// Returns the pointer to the allocated memory of size block_max_size.
    ///
    /// @param [in]   block_max_size  The required size of memory
    /// @param [in/out]  builder  The caching builder
    void* operator()(std::size_t block_max_size);

    std::shared_ptr<byte_t> result_ptr;
    std::size_t size = 0;
};
} // namespace jaccard
} // namespace oneapi::dal::preview

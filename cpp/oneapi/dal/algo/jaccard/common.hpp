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
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/detail/common.hpp"

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

class ONEAPI_DAL_EXPORT descriptor_base : public base {
public:
    using tag_t    = detail::tag;
    using float_t  = float;
    using method_t = method::by_default;

    descriptor_base();

    auto get_row_range_begin() const -> std::int64_t;
    auto get_row_range_end() const -> std::int64_t;
    auto get_column_range_begin() const -> std::int64_t;
    auto get_column_range_end() const -> std::int64_t;

protected:
    void set_row_range_impl(const int64_t& begin, const int64_t& end);
    void set_column_range_impl(const int64_t& begin, const int64_t& end);
    void set_block_impl(const std::initializer_list<int64_t>& row_range,
                        const std::initializer_list<int64_t>& column_range);

    oneapi::dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float = descriptor_base::float_t, typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using method_t = Method;

    auto& set_row_range(const int64_t& begin, const int64_t& end) {
        
        this->set_row_range_impl(begin, end);
        return *this;
    }

    auto& set_column_range(const int64_t& begin, const int64_t& end) {
        this->set_column_range_impl(begin, end);
        return *this;
    }

    auto& set_block(const std::initializer_list<int64_t>& row_range,
                    const std::initializer_list<int64_t>& column_range) {
        this->set_block_impl(row_range, column_range);
        return *this;
    }
};
} // namespace jaccard
} // namespace oneapi::dal::preview

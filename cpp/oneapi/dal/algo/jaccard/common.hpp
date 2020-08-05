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
#include "oneapi/dal/data/graph.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {
struct tag {};
class descriptor_impl;
} // namespace detail

namespace method {
struct all_vertex_pairs {};
using by_default = all_vertex_pairs;
} // namespace method

class descriptor_base : public base {
public:
    using tag_t    = detail::tag;
    using float_t  = float;
    using method_t = method::by_default;

    descriptor_base();

    auto get_row_begin() const -> std::int64_t;
    auto get_row_end() const -> std::int64_t;
    auto get_column_begin() const -> std::int64_t;
    auto get_column_end() const -> std::int64_t;

protected:
    void set_row_begin_impl(std::int64_t value);
    void set_row_end_impl(std::int64_t value);
    void set_column_begin_impl(std::int64_t value);
    void set_column_end_impl(std::int64_t value);

    oneapi::dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float = descriptor_base::float_t, typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using method_t = Method;

    auto &set_row_begin(std::int64_t value) {
        this->set_row_begin_impl(value);
        return *this;
    }

    auto &set_row_end(std::int64_t value) {
        this->set_row_end_impl(value);
        return *this;
    }

    auto &set_column_begin(std::int64_t value) {
        this->set_column_begin_impl(value);
        return *this;
    }

    auto &set_column_end(std::int64_t value) {
        this->set_column_end_impl(value);
        return *this;
    }
};
} // namespace jaccard
} // namespace oneapi::dal::preview

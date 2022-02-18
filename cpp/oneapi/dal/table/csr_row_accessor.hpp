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

#include <tuple>
#include <type_traits>

#include "oneapi/dal/table/detail/table_utils.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/backend/accessor_impl.hpp"

namespace oneapi::dal {
namespace v1 {

/// Provides access to the range of rows stored in compressed sparse rows (CSR) format.
///
/// @tparam T The type of data values in blocks returned by the accessor.
///           Should be const-qualified for read-only access. An accessor
///           supports at least :literal:`float`, :literal:`double`, and
///           :literal:`std::int32_t`.
template <typename T>
class csr_row_accessor {
    using data_t = std::remove_const_t<T>;
    static constexpr bool is_readonly = std::is_const_v<T>;
    typedef typename std::conditional<is_readonly, const std::int64_t, std::int64_t>::type I;
    using array_d = dal::array<data_t>;
    using array_i = dal::array<std::int64_t>;

public:
    /// Creates a read-only accessor object from the table. Available only for
    /// const-qualified :literal:`T`.
    template <typename U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
    explicit csr_row_accessor(const table& table) : pull_iface_(detail::get_pull_csr_block_iface(table)) {
        if (!pull_iface_) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::object_does_not_provide_read_access_to_csr() };
        }
    }

    explicit csr_row_accessor(const detail::table_builder& builder)
            : pull_iface_(detail::get_pull_csr_block_iface(static_cast<const detail::csr_table_builder&>(builder))) {
        if (!pull_iface_) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::object_does_not_provide_read_access_to_csr() };
        }
    }

    std::tuple<array_d, array_i, array_i> pull(const range& row_range = { 0, -1 },
                           const sparse_indexing indexing = sparse_indexing::one_based) const {
        array_d data;
        array_i column_indices;
        array_i row_indices;
        pull(data, column_indices, row_indices, row_range, indexing);
        return std::make_tuple(data, column_indices, row_indices);
    }

    std::tuple<T*, I*, I*> pull(array_d& data, array_i& column_indices, array_i& row_indices,
                           const range& row_range = { 0, -1 },
                           const sparse_indexing indexing = sparse_indexing::one_based) const {
        pull_iface_->pull_csr_block(detail::default_host_policy{}, data, column_indices, row_indices, indexing, row_range);
        return std::make_tuple(data_impl_.get_block_data(data),
                               indices_impl_.get_block_data(column_indices),
                               indices_impl_.get_block_data(row_indices));
    }

private:
    backend::accessor_impl<T> data_impl_;
    backend::accessor_impl<I> indices_impl_;
    detail::shared<detail::pull_csr_block_iface> pull_iface_;
};

} // namespace v1

using v1::csr_row_accessor;

} // namespace oneapi::dal
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

#include <tuple>
#include <type_traits>

#include "oneapi/dal/table/detail/table_utils.hpp"

namespace oneapi::dal::detail {
namespace v1 {

/// @tparam T The type of data values in blocks returned by the accessor.
///           Should be const-qualified for read-only access.
///           An accessor supports at least :expr:`float`, :expr:`double`, and :expr:`std::int32_t` types of :literal:`T`.
template <typename T>
class csr_accessor {
public:
    using data_t = std::remove_const_t<T>;
    using array_d = dal::array<data_t>;
    using array_i = dal::array<std::int64_t>;
    static constexpr bool is_readonly = std::is_const_v<T>;

    /// Creates a read-only accessor object from the csr table. Available only for
    /// const-qualified :literal:`T`.
    template <typename U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
    explicit csr_accessor(const csr_table& table)
            : pull_iface_(detail::get_pull_csr_block_iface(table)) {
        if (!pull_iface_) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::object_does_not_provide_read_access_to_csr() };
        }
    }

    std::tuple<array_d, array_i, array_i> pull(
        const range& row_range = { 0, -1 },
        const sparse_indexing indexing = sparse_indexing::one_based) const {
        array_d data;
        array_i column_indices;
        array_i row_offsets;
        pull_iface_->pull_csr_block(detail::default_host_policy{},
                                    data,
                                    column_indices,
                                    row_offsets,
                                    indexing,
                                    row_range);
        return std::make_tuple(data, column_indices, row_offsets);
    }

private:
    std::shared_ptr<detail::pull_csr_block_iface> pull_iface_;
};

} // namespace v1

using v1::csr_accessor;

} // namespace oneapi::dal::detail

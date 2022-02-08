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

#include "oneapi/dal/table/detail/table_utils.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

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

public:
    /// Creates a read-only accessor object from the table. Available only for
    /// const-qualified :literal:`T`.
    template <typename U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
    explicit csr_row_accessor(const table& table) : pull_iface_(detail::get_pull_rows_iface(table)) {
        if (!pull_iface_) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::object_does_not_provide_read_access_to_rows() };
        }
    }

private:
    static T* get_block_data(const dal::array<data_t>& block) {
        if constexpr (is_readonly) {
            return block.get_data();
        }
        return block.get_mutable_data();
    }

    detail::shared<detail::pull_csr_block_iface> pull_iface_;
};

} // namespace v1

using v1::csr_row_accessor;

} // namespace oneapi::dal
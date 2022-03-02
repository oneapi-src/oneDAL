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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

class row_block_info {
public:
    row_block_info() : block_index_(0), row_start_index_(0), row_count_(0), column_count_(0) {}

    std::int64_t get_start_row_index() const {
        return row_start_index_;
    }

    std::int64_t get_end_row_index() const {
        return row_start_index_ + row_count_;
    }

    range get_row_range() const {
        return { get_start_row_index(), get_end_row_index() };
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    std::int64_t get_column_count() const {
        return column_count_;
    }

    ndshape<2> get_shape() const {
        return { row_count_, column_count_ };
    }

    std::int64_t get_block_index() const {
        return block_index_;
    }

    const row_block_info& update(std::int64_t block_index,
                                 std::int64_t row_start_index,
                                 std::int64_t row_count,
                                 std::int64_t column_count) {
        block_index_ = block_index;
        row_start_index_ = row_start_index;
        row_count_ = row_count;
        column_count_ = column_count;
        return *this;
    }

private:
    std::int64_t block_index_;
    std::int64_t row_start_index_;
    std::int64_t row_count_;
    std::int64_t column_count_;
};

/// Helper function that simplifies looping over the blocked data.
/// See detailed description below.
template <typename T, typename Body>
inline void for_each_block(const ndview<T, 2>& data,
                           std::int64_t block_max_row_count,
                           Body&& body) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(block_max_row_count > 0);

    for_each_block(data.get_dimension(0),
                   data.get_dimension(1),
                   block_max_row_count,
                   std::forward<Body>(body));
}

/// Helper function that simplifies looping over the blocked data.
/// See detailed description below.
template <typename Body>
inline void for_each_block(std::int64_t row_count,
                           std::int64_t column_count,
                           std::int64_t block_max_row_count,
                           Body&& body) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(block_max_row_count > 0);

    const std::int64_t block_count = row_count / block_max_row_count;
    const std::int64_t tail_block_row_count = row_count % block_max_row_count;

    row_block_info info;

    for (std::int64_t i = 0; i < block_count; i++) {
        body(info.update(i, i * block_max_row_count, block_max_row_count, column_count));
    }

    if (tail_block_row_count > 0) {
        const std::int64_t i = block_count;
        body(info.update(i, i * block_max_row_count, tail_block_row_count, column_count));
    }
}

/// Helper function that simplifies looping over the blocked data
///
/// Example of recommended usage:
/// @code
/// array<T> block_flat;
/// const auto acc = row_accessor<const T>{ x };
/// const std::int64_t block_row_count = 2048;
///
/// for_each_block(x, block_row_count, [&](const row_block_info& bi) mutable {
///     const T* block_ptr = acc.pull(queue, block_flat, bi.get_range());
///     const auto block = ndview<T, 2>::wrap(block_ptr, bi.get_shape());
/// });
/// @endcode
///
/// @tparam Body The user's block handler, must be a functor that accepts `row_block_info`
///
/// @param data                The data needs to be blocked
/// @param block_max_row_count The maximal row count in each block. `body` is not
///                            guarantied to be called with the provided `block_max_row_count`.
///                            The "tail" block (if data row count is not mutiple of
///                            `block_max_row_count`) always contains less rows.
/// @param body                The user-provided lambda
template <typename Body>
inline void for_each_block(const table& data, std::int64_t block_max_row_count, Body&& body) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(block_max_row_count > 0);

    for_each_block(data.get_row_count(),
                   data.get_column_count(),
                   block_max_row_count,
                   std::forward<Body>(body));
}

} // namespace oneapi::dal::backend::primitives

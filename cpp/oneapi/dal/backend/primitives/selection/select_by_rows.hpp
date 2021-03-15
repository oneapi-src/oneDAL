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

#include <type_traits>

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
class selection_by_rows {
public:
    selection_by_rows() = delete;

    /// Constructor for selection on every row of matrix
    ///
    /// @param[in]  data The [n x m] input matrix
    explicit selection_by_rows(const ndview<Float, 2>& data)
            : data_(data),
              /*add_by_col_(dummy_array_),*/ col_begin_(0),
              col_end_(data.get_dimension(1)) {
        ONEDAL_ASSERT(data_.get_count() > 0);
    }
    /// Constructor for selection in range [col_begin, col_end) on every row
    ///
    /// @param[in]  data The [n x m] input matrix
    /// @param[in]  col_begin_ The start col for selection over row (0 <= col_begin < data.get_column_count())
    /// @param[in]  col_end The start col for selection over row (col_begin < col_end <= data.get_column_count())
    explicit selection_by_rows(const ndview<Float, 2>& data,
                               std::int64_t col_begin,
                               std::int64_t col_end)
            : data_(data),
              /*add_by_col_(dummy_array_),*/ col_begin_(col_begin),
              col_end_(col_end) {
        ONEDAL_ASSERT(data_.get_count() > 0);
        ONEDAL_ASSERT(col_begin >= 0);
        ONEDAL_ASSERT(col_begin < data.get_column_count());
        ONEDAL_ASSERT(col_end > 0);
        ONEDAL_ASSERT(col_end <= data.get_column_count());
    }
    /*
New functionality for kNN performance improvement (optimized usage of GEMM)

/// Constructor for selection in range [col_begin, col_end) with additional array added to sum with the range in every row
///
/// @param[in]  data The [n x m] input matrix
/// @param[in]  add_by_col The [1, (col_end - col_begin)] input array
/// @param[in]  col_begin_ The start col for selection over row (0 <= col_begin < data.get_column_count())
/// @param[in]  col_end The start col for selection over row (col_begin < col_end <= data.get_column_count())
    explicit selection_by_rows(const ndview<Float, 2>& data, const ndview<Float, 1>& add_by_col, std::int64_t col_begin, std::int64_t last_col)
            : data_(data), add_by_col_(add_by_col), col_begin_(col_begin), col_end_(col_end) {
        ONEDAL_ASSERT(data_.get_count() > 0);
        ONEDAL_ASSERT(col_begin >= 0);
        ONEDAL_ASSERT(col_begin < data.get_column_count());
        ONEDAL_ASSERT(col_end > 0);
        ONEDAL_ASSERT(col_end < data.get_column_count());
        ONEDAL_ASSERT(add_by_col_.get_column_count() == col_end_ - col_begin_);
    }

/// Constructor for selection  with additional array added to sum with every row
///
/// @param[in]  data The [n x m] input matrix
/// @param[in]  add_by_col The [1, m] input array
    explicit selection_by_rows(const ndview<Float, 2>& data, const ndview<Float, 1>& add_by_col)
            : data_(data), add_by_col_(add_by_col), col_begin_(col_begin), col_end_(col_end) {
        ONEDAL_ASSERT(data_.get_count() > 0);
        ONEDAL_ASSERT(col_begin >= 0);
        ONEDAL_ASSERT(col_begin < data.get_column_count());
        ONEDAL_ASSERT(col_end > 0);
        ONEDAL_ASSERT(col_end < data.get_column_count());
        ONEDAL_ASSERT(add_by_col_.get_column_count() == col_end_ - col_begin_);
    }
*/
    /// Performs K-selection on each row in matrix
    ///
    /// @param[in]  queue The queue
    /// @param[in]  k      The number of minimal values to be selected in each row
    /// @param[out] selection The [n x k] matrix of selected values (if selected_out == true)
    /// @param[out] indices  The [n x k] matrix of indices of selected values (if indices_out == true)

    sycl::event select(sycl::queue& queue,
                       std::int64_t k,
                       ndview<Float, 2>& selection,
                       ndview<int, 2>& column_indices,
                       const event_vector& deps = {});

    /// Performs K-selection on each row in matrix
    ///
    /// @param[in]  queue The queue
    /// @param[in]  k      The number of minimal values to be selected in each row
    /// @param[out] selection The [n x k] matrix of selected values (if selected_out == true)

    sycl::event select(sycl::queue& queue,
                       std::int64_t k,
                       ndview<Float, 2>& selection,
                       const event_vector& deps = {});

    /// Performs K-selection on each row in matrix
    ///
    /// @param[in]  queue The queue
    /// @param[in]  k      The number of minimal values to be selected in each row
    /// @param[out] column_indices  The [n x k] matrix of indices of selected values (if indices_out == true)

    sycl::event select(sycl::queue& queue,
                       std::int64_t k,
                       ndview<int, 2>& column_indices,
                       const event_vector& deps = {});

private:
    const ndview<Float, 2>& data_;
    /*    const ndview<Float, 1>& add_by_col;*/
    std::int64_t col_begin_;
    std::int64_t col_end_;
    /*    static ndview<Float, 1> dummy_array_;*/
};

#endif

} // namespace oneapi::dal::backend::primitives

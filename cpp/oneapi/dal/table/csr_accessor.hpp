/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/dal/table/csr.hpp"

namespace oneapi::dal {
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
    typedef typename std::conditional<is_readonly, const std::int64_t, std::int64_t>::type I;

    /// Creates a read-only accessor object from the csr table. Available only for
    /// const-qualified :literal:`T`.
    ///
    /// @tparam U The type of data values in blocks returned by the accessor.
    ///           Should be const-qualified for read-only access.
    ///           An accessor supports at least :expr:`float`, :expr:`double`, and :expr:`std::int32_t` types of :literal:`U`.
    ///
    /// @param table Input CSR table.
    template <typename U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
    explicit csr_accessor(const csr_table& table)
            : pull_iface_(detail::get_pull_csr_block_iface(table)) {
        if (!pull_iface_) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::object_does_not_provide_read_access_to_csr() };
        }
    }

    /// Provides access to the rows of the table in CSR format
    /// The method returns arrays that directly point to the memory within the table
    /// if it is possible. In that case, the arrays refer to the memory as immutable data.
    /// Otherwise, new memory blocks are allocated, and the data from the table rows is converted
    /// and copied into those blocks. In this case, arrays refer to the blocks as mutable data.
    ///
    /// @param[in] row_range The range of rows that data is returned from the accessor.
    /// @param[in] indexing  The indexing scheme used to access the data in the returned arrays
    ///                      in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
    ///                      :literal:`sparse_indexing::one_based`.
    ///
    /// @return A tuple of three arrays: values, column indicies, and row offsets that represent a sub-table
    ///         in CSR format that contain the data from the original table corresponding to the rows from the `row_range`
    ///         and with the requested indexing scheme.
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

#ifdef ONEDAL_DATA_PARALLEL
    /// Provides access to the rows of the table in CSR format
    /// The method returns arrays that directly point to the memory within the table
    /// if it is possible. In that case, the arrays refer to the memory as immutable data.
    /// Otherwise, new memory blocks are allocated, and the data from the table rows is converted
    /// and copied into those blocks. In this case, arrays refer to the blocks as mutable data.
    ///
    /// @param[in] queue     The SYCL* queue object.
    /// @param[in] row_range The range of rows that data is returned from the accessor.
    /// @param[in] indexing  The indexing scheme used to access the data in the returned arrays
    ///                      in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
    ///                      :literal:`sparse_indexing::one_based`.
    /// @param[in] alloc     The requested kind of USM in the returned block.
    ///
    /// @return A tuple of three arrays: values, column indicies, and row offsets that represent a sub-table
    ///         in CSR format that contain the data from the original table corresponding to the rows from the `row_range`
    ///         and with the requested indexing scheme.
    std::tuple<array_d, array_i, array_i> pull(
        sycl::queue& queue,
        const range& row_range = { 0, -1 },
        const sparse_indexing indexing = sparse_indexing::one_based,
        const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        array_d data;
        array_i column_indices;
        array_i row_offsets;
        pull_iface_->pull_csr_block(detail::data_parallel_policy{ queue },
                                    data,
                                    column_indices,
                                    row_offsets,
                                    indexing,
                                    row_range,
                                    alloc);
        return std::make_tuple(data, column_indices, row_offsets);
    }
#endif

    /// Provides access to the rows of the table in CSR format
    /// The method returns arrays that directly point to the memory within the table
    /// if it is possible. In that case, the arrays refer to the memory as immutable data.
    /// Otherwise, new memory blocks are allocated, and the data from the table rows is converted
    /// and copied into those blocks. In this case, arrays refer to the blocks as mutable data.
    /// The method updates :expr:`data`, :expr:`column_indices`, and :expr:`row_offsets` arrays.
    ///
    /// @param[in,out] data           The block in which memory is reused (if it is possible) to obtain
    ///                               the values from the table.
    /// @param[in,out] column_indices The block in which memory is reused (if it is possible) to obtain
    ///                               the column indices from the table.
    /// @param[in,out] row_offsets    The block in which memory is reused (if it is possible) to obtain
    ///                               the row offsets from the table.
    ///                               The memory of `data`, `column_indices` and `row_offsets` blocks
    ///                               are reset either when their size is not big enough, or when the blocks
    ///                               contain immutable data, or when direct memory from the table
    ///                               can be used. If the blocks are reset to use direct memory pointers
    ///                               from the object, they refer to those pointers as immutable memory
    ///                               blocks.
    /// @param[in] row_range          The range of rows that data is returned from the accessor.
    /// @param[in] indexing           The indexing scheme used to access the data in the returned arrays
    ///                               in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
    ///                               :literal:`sparse_indexing::one_based`.
    ///
    /// @return A tuple of three pointers: values, column indicies, and row offsets that represent
    ///         a sub-table in CSR format that contain the data from the original table corresponding
    ///         to the rows from the `row_range` and with the requested indexing scheme.
    std::tuple<T*, I*, I*> pull(array_d& data,
                                array_i& column_indices,
                                array_i& row_offsets,
                                const range& row_range = { 0, -1 },
                                const sparse_indexing indexing = sparse_indexing::one_based) const {
        pull_iface_->pull_csr_block(detail::default_host_policy{},
                                    data,
                                    column_indices,
                                    row_offsets,
                                    indexing,
                                    row_range);
        return get_block_data(data, column_indices, row_offsets);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Provides access to the rows of the table in CSR format
    /// The method returns arrays that directly point to the memory within the table
    /// if it is possible. In that case, the arrays refer to the memory as immutable data.
    /// Otherwise, new memory blocks are allocated, and the data from the table rows is converted
    /// and copied into those blocks. In this case, arrays refer to the blocks as mutable data.
    /// The method updates :expr:`data`, :expr:`column_indices`, and :expr:`row_offsets` arrays.
    ///
    /// @param[in] queue              The SYCL* queue object.
    /// @param[in,out] data           The block in which memory is reused (if it is possible) to obtain
    ///                               the values from the table.
    /// @param[in,out] column_indices The block in which memory is reused (if it is possible) to obtain
    ///                               the column indices from the table.
    /// @param[in,out] row_offsets    The block in which memory is reused (if it is possible) to obtain
    ///                               the row offsets from the table.
    ///                               The memory of `data`, `column_indices`, and `row_offsets` blocks
    ///                               are reset either when their size is not big enough, or when the blocks
    ///                               contain immutable data, or when direct memory from the table
    ///                               can be used. If the blocks are reset to use direct memory pointers
    ///                               from the object, they refer to those pointers as immutable memory
    ///                               blocks.
    /// @param[in] row_range          The range of rows that data is returned from the accessor.
    /// @param[in] indexing           The indexing scheme used to access the data in the returned arrays
    ///                               in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
    ///                               :literal:`sparse_indexing::one_based`.
    /// @param[in] alloc              The requested kind of USM in the returned block.
    ///
    /// @return A tuple of three pointers: values, column indicies, and row offsets that represent
    ///         a sub-table in CSR format that contain the data from the original table corresponding
    ///         to the rows from the `row_range` and with the requested indexing scheme.
    std::tuple<T*, I*, I*> pull(sycl::queue& queue,
                                array_d& data,
                                array_i& column_indices,
                                array_i& row_offsets,
                                const range& row_range = { 0, -1 },
                                const sparse_indexing indexing = sparse_indexing::one_based,
                                const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        pull_iface_->pull_csr_block(detail::data_parallel_policy{ queue },
                                    data,
                                    column_indices,
                                    row_offsets,
                                    indexing,
                                    row_range,
                                    alloc);
        return get_block_data(data, column_indices, row_offsets);
    }
#endif

private:
    static std::tuple<T*, I*, I*> get_block_data(const array_d& data,
                                                 array_i& column_indices,
                                                 array_i& row_offsets) {
        if constexpr (is_readonly) {
            return std::make_tuple(data.get_data(),
                                   column_indices.get_data(),
                                   row_offsets.get_data());
        }
        return std::make_tuple(data.get_mutable_data(),
                               column_indices.get_mutable_data(),
                               row_offsets.get_mutable_data());
    }

    std::shared_ptr<detail::pull_csr_block_iface> pull_iface_;
};

} // namespace v1

using v1::csr_accessor;

} // namespace oneapi::dal

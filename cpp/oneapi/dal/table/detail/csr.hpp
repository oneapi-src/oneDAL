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
#include "oneapi/dal/detail/array_utils.hpp"
#include "oneapi/dal/table/detail/sparse_access_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class ONEDAL_EXPORT csr_table : public table {
    friend detail::pimpl_accessor;

public:
    /// Returns the unique id of ``csr_table`` class.
    static std::int64_t kind();

    /// Creates a new ``csr_table`` instance with zero number of rows and columns.
    /// The :expr:`kind` is set to``csr_table::kind()``.
    /// All the properties should be set to default values (see the Properties section).
    csr_table();

    /// Creates a new ``csr_table`` instance from externally-defined data blocks.
    /// Table object owns the data, row indices and column indices pointers.
    ///
    /// @tparam Data          The type of elements in the data block that will be stored into the table.
    ///                       The :literal:`Data` type should be at least :expr:`float`, :expr:`double` or :expr:`std::int32_t`.
    /// @tparam DataDeleter   The type of a deleter called on ``data`` when
    ///                       the last table that refers it is out of the scope.
    /// @tparam ColumnDeleter The type of a deleter called on ``column_indices`` when
    ///                       the last table that refers it is out of the scope.
    /// @tparam RowDeleter    The type of a deleter called on ``row_indices`` when
    ///                       the last table that refers it is out of the scope.
    ///
    /// @param data           The pointer to values in the CSR layout.
    /// @param column_indices The pointer to column indices in the CSR layout.
    /// @param row_indices    The pointer to row indices in the CSR layout.
    /// @param row_count      The number of rows in the corresponding dense table.
    /// @param column_count   The number of columns in the corresponding dense table.
    /// @param data_deleter   The deleter that is called on the ``data`` when the last table that refers it
    ///                       is out of the scope.
    /// @param column_deleter The deleter that is called on the ``column_indices`` when the last table that refers it
    ///                       is out of the scope.
    /// @param row_deleter    The deleter that is called on the ``row_indices`` when the last table that refers it
    ///                       is out of the scope.
    /// @param indexing       The indexing scheme used to access data in the CSR layout. Support only :literal:`csr_indexing::one_based`.
    template <typename Data, typename DataDeleter, typename ColumnDeleter, typename RowDeleter>
    csr_table(const Data* data,
              const std::int64_t* column_indices,
              const std::int64_t* row_indices,
              std::int64_t row_count,
              std::int64_t column_count,
              DataDeleter&& data_deleter,
              ColumnDeleter&& column_deleter,
              RowDeleter&& row_deleter,
              csr_indexing indexing = csr_indexing::one_based) {
        init_impl(detail::default_host_policy{},
                  column_count,
                  row_count,
                  data,
                  column_indices,
                  row_indices,
                  std::forward<DataDeleter>(data_deleter),
                  std::forward<ColumnDeleter>(column_deleter),
                  std::forward<RowDeleter>(row_deleter),
                  indexing);
    }

    /// The unique id of the csr table type.
    std::int64_t get_kind() const {
        return kind();
    }

    /// Returns the :literal:`data` pointer cast to the :literal:`Data` type. No checks are
    /// performed that this type is the actual type of the data within the table.
    template <typename Data>
    const Data* get_data() const {
        return reinterpret_cast<const Data*>(this->get_data());
    }

    /// The pointer to the data block within the table.
    /// Should be equal to ``nullptr`` when :expr:`row_count == 0` and :expr:`column_count == 0`.
    const void* get_data() const;

    /// Returns the :literal:`column_indices` pointer.
    const std::int64_t* get_column_indices() const;

    /// Returns the :literal:`row_indices` pointer.
    const std::int64_t* get_row_indices() const;

private:
    explicit csr_table(detail::csr_table_iface* impl) : table(impl) {}

    bool correct_indices(const std::int64_t size,
                         const std::int64_t* indices,
                         const csr_indexing indexing) const {
        const std::int64_t min_value = (indexing == csr_indexing::zero_based) ? 0 : 1;
        for (std::int64_t i = 0; i < size; i++) {
            if (indices[i] < min_value) {
                return false;
            }
        }
        return true;
    }

    template <typename Policy,
              typename Data,
              typename DataDeleter,
              typename ColumnDeleter,
              typename RowDeleter>
    void init_impl(const Policy& policy,
                   std::int64_t column_count,
                   std::int64_t row_count,
                   const Data* data,
                   const std::int64_t* column_indices,
                   const std::int64_t* row_indices,
                   DataDeleter&& data_deleter,
                   ColumnDeleter&& column_deleter,
                   RowDeleter&& row_deleter,
                   csr_indexing indexing) {
        using error_msg = dal::detail::error_messages;

        if (row_count <= 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        if (!correct_indices(column_count, column_indices, indexing) ||
            !correct_indices(row_count, row_indices, indexing)) {
            throw dal::domain_error(error_msg::invalid_indices());
        }

        array<Data> data_array{ data,
                                row_indices[row_count] - row_indices[0],
                                std::forward<DataDeleter>(data_deleter) };

        array<std::int64_t> column_indices_array{ column_indices,
                                                  row_indices[row_count] - row_indices[0],
                                                  std::forward<ColumnDeleter>(column_deleter) };

        array<std::int64_t> row_indices_array{ row_indices,
                                               row_count + 1,
                                               std::forward<RowDeleter>(row_deleter) };

        auto byte_data = reinterpret_cast<const byte_t*>(data);
        dal::detail::check_mul_overflow(data_array.get_count(),
                                        static_cast<std::int64_t>(sizeof(Data)));
        const std::int64_t byte_count =
            data_array.get_count() * static_cast<std::int64_t>(sizeof(Data));

        auto byte_array = array<byte_t>{ data_array, byte_data, byte_count };

        init_impl(policy,
                  column_count,
                  row_count,
                  byte_array,
                  column_indices_array,
                  row_indices_array,
                  detail::make_data_type<Data>(),
                  indexing);
    }

    template <typename Policy>
    void init_impl(const Policy& policy,
                   std::int64_t column_count,
                   std::int64_t row_count,
                   const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_indices,
                   const data_type& dtype,
                   csr_indexing indexing);
};

} // namespace v1

using v1::csr_table;

} // namespace oneapi::dal::detail

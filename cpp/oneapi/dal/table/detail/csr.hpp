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
#include "oneapi/dal/table/detail/csr_block.hpp"
#include "oneapi/dal/detail/array_utils.hpp"

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
    ///
    /// @param data           The array of values in the CSR layout.
    /// @param column_indices The array of column indices in the CSR layout.
    /// @param row_indices    The array of row indices in the CSR layout.
    /// @param row_count      The number of rows in the corresponding dense table.
    /// @param column_count   The number of columns in the corresponding dense table.
    /// @param indexing       The indexing scheme used to access data in the CSR layout. Support only :literal:`csr_indexing::one_based`.
    template <typename Data>
    csr_table(const dal::array<Data>& data,
              const dal::array<std::int64_t>& column_indices,
              const dal::array<std::int64_t>& row_indices,
              std::int64_t row_count,
              std::int64_t column_count,
              csr_indexing indexing = csr_indexing::one_based) {
        init_impl(data, column_indices, row_indices, row_count, column_count, indexing);
    }

    /// The unique id of the csr table type.
    std::int64_t get_kind() const {
        return kind();
    }

    /// The number of non-zero elements in the table.
    /// @remark default = 0
    std::int64_t get_non_zero_count() const;

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

    void check_indices(const std::int64_t row_count,
                       const std::int64_t column_count,
                       const std::int64_t* row_indices,
                       const std::int64_t* column_indices,
                       const csr_indexing indexing) const {
        using error_msg = dal::detail::error_messages;
        const std::int64_t min_value = (indexing == csr_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_value =
            (indexing == csr_indexing::zero_based) ? column_count - 1 : column_count;
        const std::int64_t element_count = row_indices[row_count] - row_indices[0];
        const std::int64_t last_value = row_indices[row_count];

        if (row_count <= 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        if (indexing != csr_indexing::one_based) {
            throw dal::domain_error(detail::error_messages::zero_based_indexing_is_not_supported());
        }

        if (element_count < 0) {
            throw dal::domain_error(error_msg::row_indices_lt_min_value());
        }

        for (std::int64_t i = 0; i <= row_count; i++) {
            if (row_indices[i] < min_value) {
                throw dal::domain_error(error_msg::row_indices_lt_min_value());
            }
            if (row_indices[i] > last_value) {
                throw dal::domain_error(error_msg::row_indices_gt_max_value());
            }
        }

        for (std::int64_t i = 0; i < element_count; i++) {
            if (column_indices[i] < min_value) {
                throw dal::domain_error(error_msg::column_indices_lt_min_value());
            }
            if (column_indices[i] > max_value) {
                throw dal::domain_error(error_msg::column_indices_gt_max_value());
            }
        }
    }

    template <typename Data>
    void init_impl(const dal::array<Data>& data,
                   const dal::array<std::int64_t>& column_indices,
                   const dal::array<std::int64_t>& row_indices,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   csr_indexing indexing) {
        check_indices(row_count,
                      column_count,
                      row_indices.get_data(),
                      column_indices.get_data(),
                      indexing);

        init_impl(column_count,
                  row_count,
                  detail::reinterpret_array_cast<byte_t>(data),
                  column_indices,
                  row_indices,
                  detail::make_data_type<Data>(),
                  indexing);
    }

    void init_impl(std::int64_t column_count,
                   std::int64_t row_count,
                   const dal::array<byte_t>& data,
                   const dal::array<std::int64_t>& column_indices,
                   const dal::array<std::int64_t>& row_indices,
                   const data_type& dtype,
                   csr_indexing indexing);
};

} // namespace v1

using v1::csr_table;

} // namespace oneapi::dal::detail

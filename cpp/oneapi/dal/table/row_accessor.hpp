/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal {
namespace v1 {

/// @tparam T The type of data values in blocks returned by the accessor.
///           Should be const-qualified for read-only access.
///           An accessor supports at least :expr:`float`, :expr:`double`, and :expr:`std::int32_t` types of :literal:`T`.
template <typename T>
class row_accessor {
public:
    using data_t = std::remove_const_t<T>;
    static constexpr bool is_readonly = std::is_const_v<T>;

    /// Creates a new read-only accessor object from the table.
    /// The check that the accessor supports the table kind of :literal:`obj` is performed.
    /// The reference to the :literal:`obj` table is stored within the accessor to
    /// obtain data from the table.
    explicit row_accessor(const table& table) : pull_iface_(detail::get_pull_rows_iface(table)) {
        static_assert(is_readonly,
                      "Tables can be used only for pull operations, "
                      "use row_accessor<const T> instead");

        if (!pull_iface_) {
            // TODO: Replace to error_messages
            throw invalid_argument{ "Given table does not provide read access to rows" };
        }
    }

    explicit row_accessor(const detail::table_builder& builder)
            : pull_iface_(detail::get_pull_rows_iface(builder)),
              push_iface_(detail::get_push_rows_iface(builder)) {
        if (!pull_iface_) {
            // TODO: Replace to error_messages
            throw invalid_argument{ "Given table builder does not provide read access to rows" };
        }

        if (!is_readonly && !push_iface_) {
            // TODO: Replace to error_messages
            throw invalid_argument{ "Given table builder does not provide write access to rows" };
        }
    }

    array<data_t> pull(const range& rows = { 0, -1 }) const {
        array<data_t> block;
        pull(block, rows);
        return block;
    }

    T* pull(array<data_t>& block, const range& rows = { 0, -1 }) const {
        pull_iface_->pull_rows(detail::default_host_policy{}, block, rows);
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Provides access to the rows of the table.
    /// The method returns an array that directly points to the memory within the table
    /// if it is possible. In that case, the array refers to the memory as to immutable data.
    /// Otherwise, the new memory block is allocated, the data from the table rows is converted
    /// and copied into this block. The array refers to the block as to mutable data.
    ///
    /// @param[in] queue The SYCL* queue object.
    /// @param[in] rows  The range of rows that data is returned from the accessor.
    /// @param[in] alloc The requested kind of USM in the returned block.
    ///
    /// @pre ``rows`` are within the range of ``[0, obj.row_count)``.
    array<data_t> pull(sycl::queue& queue,
                       const range& rows = { 0, -1 },
                       const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        array<data_t> block;
        pull(queue, block, rows, alloc);
        return block;
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    /// Provides access to the rows of the table.
    /// The method returns the :expr:`block.data` pointer.
    ///
    /// @param[in] queue     The SYCL* queue object.
    /// @param[in,out] block The block which memory is reused (if it is possible) to obtain the data from the table.
    ///                      The block memory is reset either when
    ///                      its size is not big enough, or when it contains immutable data, or when direct
    ///                      memory from the table can be used.
    ///                      If the block is reset to use a direct memory pointer from the object,
    ///                      it refers to this pointer as to immutable memory block.
    /// @param[in] rows      The range of rows that data is returned from the accessor.
    /// @param[in] alloc     The requested kind of USM in the returned block.
    ///
    /// @pre ``rows`` are within the range of ``[0, obj.row_count)``.
    T* pull(sycl::queue& queue,
            array<data_t>& block,
            const range& rows = { 0, -1 },
            const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        pull_iface_->pull_rows(detail::data_parallel_policy{ queue }, block, rows, alloc);
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }
#endif

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        push_iface_->push_rows(detail::default_host_policy{}, block, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(sycl::queue& queue,
                                                     const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        push_iface_->push_rows(detail::data_parallel_policy{ queue }, block, rows);
    }
#endif

private:
    std::shared_ptr<detail::pull_rows_iface> pull_iface_;
    std::shared_ptr<detail::push_rows_iface> push_iface_;
};

} // namespace v1

using v1::row_accessor;

} // namespace oneapi::dal

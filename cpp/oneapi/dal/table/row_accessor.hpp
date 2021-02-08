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

#include "oneapi/dal/table/detail/accessor_base.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal {
namespace v1 {

/// @tparam T The type of data values in blocks returned by the accessor.
///           Should be const-qualified for read-only access.
///           An accessor supports at least :expr:`float`, :expr:`double`, and :expr:`std::int32_t` types of :literal:`T`.
template <typename T>
class row_accessor : private detail::accessor_base<T, detail::row_block> {
    using base = detail::accessor_base<T, detail::row_block>;

public:
    using data_t = typename base::data_t;
    static constexpr bool is_readonly = base::is_readonly;

    /// Creates a new read-only accessor object from the table.
    /// The check that the accessor supports the table kind of :literal:`obj` is performed.
    /// The reference to the :literal:`obj` table is stored within the accessor to
    /// obtain data from the table.
    template <
        typename K,
        typename = std::enable_if_t<is_readonly && (std::is_base_of_v<table, K> ||
                                                    std::is_base_of_v<detail::table_builder, K>)>>
    row_accessor(const K& obj) : base(obj) {}

    row_accessor(const detail::table_builder& b) : base(b) {}

    array<data_t> pull(const range& rows = { 0, -1 }) const {
        return base::pull(detail::default_host_policy{},
                          { rows },
                          detail::host_allocator<data_t>{});
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
        return base::pull(detail::data_parallel_policy{ queue },
                          { rows },
                          detail::data_parallel_allocator<data_t>(queue, alloc));
    }
#endif

    T* pull(array<data_t>& block, const range& rows = { 0, -1 }) const {
        return base::pull(detail::default_host_policy{},
                          block,
                          { rows },
                          detail::host_allocator<data_t>{});
    }

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
        return base::pull(detail::data_parallel_policy{ queue },
                          block,
                          { rows },
                          detail::data_parallel_allocator<data_t>(queue, alloc));
    }
#endif

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::default_host_policy{}, block, { rows });
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(sycl::queue& queue,
                                                     const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::data_parallel_policy{ queue }, block, { rows });
    }
#endif
};

} // namespace v1

using v1::row_accessor;

} // namespace oneapi::dal

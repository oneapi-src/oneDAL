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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

namespace oneapi::dal::detail {
namespace v1 {

/// Builds data table from the provided data.
class ONEDAL_EXPORT table_builder {
    friend pimpl_accessor;

public:
    /// Build data table.
    table build() const {
        return detail::make_private<table>(impl_->build());
    }

protected:
    explicit table_builder(table_builder_iface* impl) : impl_(impl) {}

private:
    pimpl<table_builder_iface> impl_;
};

/// Builds homogeneous table from the provided data.
class ONEDAL_EXPORT homogen_table_builder : public table_builder {
public:
    homogen_table_builder();

    /// Build homogeneous table.
    homogen_table build() {
        return detail::make_private<homogen_table>(get_impl().build_homogen());
    }

    /// Reset the internal data of the builder with the data from the input table.
    ///
    /// @param t    R-value reference to input homogeneous table.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& reset(homogen_table&& t) {
        const homogen_table local_table = std::move(t);

        const std::int64_t row_count = local_table.get_row_count();
        const std::int64_t column_count = local_table.get_column_count();
        const data_type dtype = local_table.get_metadata().get_data_type(0);
        const auto byte_data = get_original_data(local_table);

        get_impl().set_data_type(dtype);
        get_impl().reset(byte_data, row_count, column_count);
        return *this;
    }

    /// Reset the internal data of the builder with the provided inputs.
    ///
    /// @tparam Data    The type of elements in the input data block.
    ///                 The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                 or :expr:`std::int32_t`.
    ///
    /// @param data         The array that stores a homogeneous data block.
    /// @param row_count    The number of rows in the table to be built.
    /// @param column_count The number of columns in the table to be built.
    ///
    /// @result Reference to the updated homogeneous table builder.
    template <typename Data>
    auto& reset(const dal::array<Data>& data, std::int64_t row_count, std::int64_t column_count) {
        const auto byte_data = detail::reinterpret_array_cast<byte_t>(data);
        get_impl().set_data_type(detail::make_data_type<Data>());
        get_impl().reset(byte_data, row_count, column_count);
        return *this;
    }

    /// Reset the internal data of the builder with the provided inputs.
    ///
    /// @param data         The array that stores a homogeneous data block.
    /// @param row_count    The number of rows in the table to be built.
    /// @param column_count The number of columns in the table to be built.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& set_data_type(data_type dt) {
        get_impl().set_data_type(dt);
        return *this;
    }

    /// Set the type of the features in the data table
    ///
    /// @param ft The type of the features. Should be :literal:`feature_type::nominal`,
    ///           :literal:`feature_type::ordinal`, :literal:`feature_type::interval` or
    ///           :literal:`feature_type::ratio`.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& set_feature_type(feature_type ft) {
        get_impl().set_feature_type(ft);
        return *this;
    }

    /// Set the type of the layout of the data in the data table
    ///
    /// @param layout   The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                 :literal:`data_layout::column_major`.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& set_layout(data_layout layout) {
        get_impl().set_layout(layout);
        return *this;
    }

    /// Allocate the storage for the data table.
    ///
    /// @param row_count    The number of rows in the data block to allocate.
    /// @param column_count The number of columns in the data block to allocate.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& allocate(std::int64_t row_count, std::int64_t column_count) {
        get_impl().allocate(row_count, column_count);
        return *this;
    }

    /// Copy data from the data block into the table builder.
    ///
    /// @tparam Data    The type of elements in the input data block.
    ///                 The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                 or :expr:`std::int32_t`.
    ///
    /// @param data         The pointer to a homogeneous data block.
    /// @param row_count    The number of rows in the data block.
    /// @param column_count The number of columns in the data blockt.
    ///
    /// @result Reference to the updated homogeneous table builder.
    template <typename Data>
    auto& copy_data(const Data* data, std::int64_t row_count, std::int64_t column_count) {
        get_impl().copy_data(data, row_count, column_count);
        return *this;
    }

    /// Copy data from the array.
    ///
    /// @tparam Data         The type of elements in the input array.
    ///                      The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                      or :expr:`std::int32_t`.
    ///
    /// @param data         The array that stores a homogeneous data block.
    ///
    /// @result Reference to the updated homogeneous table builder.
    template <typename Data>
    auto& copy_data(const array<Data>& data) {
        get_impl().copy_data(reinterpret_array_cast<dal::byte_t>(data));
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Allocate the USM storage for the data table.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param row_count    The number of rows in the data block to allocate.
    /// @param column_count The number of columns in the data block to allocate.
    /// @param alloc        The requested kind of USM in the allocated block.
    ///
    /// @result Reference to the updated homogeneous table builder.
    auto& allocate(const sycl::queue& queue,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        get_impl().allocate(detail::data_parallel_policy{ queue }, row_count, column_count, alloc);
        return *this;
    }

    /// Copy data from the USM data block into the table builder.
    ///
    /// @tparam Data    The type of elements in the input data block.
    ///                 The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                 or :expr:`std::int32_t`.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The pointer to a homogeneous data block.
    /// @param row_count    The number of rows in the data block.
    /// @param column_count The number of columns in the data block.
    /// @param dependencies Events indicating availability of the :literal:`data` for reading or writing.
    ///
    /// @result Reference to the updated homogeneous table builder.
    template <typename Data>
    auto& copy_data(sycl::queue& queue,
                    const Data* data,
                    std::int64_t row_count,
                    std::int64_t column_count,
                    const std::vector<sycl::event>& dependencies = {}) {
        sycl::event::wait_and_throw(dependencies);
        get_impl().copy_data(detail::data_parallel_policy{ queue }, data, row_count, column_count);
        return *this;
    }
#endif

private:
    homogen_table_builder_iface& get_impl() {
        return cast_impl<homogen_table_builder_iface>(*this);
    }
};

/// Builds compressed sparse rows (CSR) table
class ONEDAL_EXPORT csr_table_builder : public table_builder {
public:
    csr_table_builder();

    /// Build CSR table.
    csr_table build() {
        return detail::make_private<csr_table>(get_impl().build_csr());
    }

    /// Reset the internal data of the builder with the provided inputs.
    ///
    /// @tparam Data    The type of elements in the input values block.
    ///                 The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                 or :expr:`std::int32_t`.
    ///
    /// @param data           The array of values in the CSR layout.
    /// @param column_indices The array of column indices in the CSR layout.
    /// @param row_offsets    The array of row offsets in the CSR layout.
    /// @param row_count      The number of rows in the table to be built.
    /// @param column_count   The number of columns in the table to be built.
    /// @param indexing       The indexing scheme used to access data in the CSR layout.
    ///                       Should be :literal:`sparse_indexing::zero_based` or
    ///                       :literal:`sparse_indexing::one_based`.
    ///
    /// @result Reference to the updated CSR table builder.
    template <typename Data>
    auto& reset(const dal::array<Data>& data,
                const dal::array<std::int64_t>& column_indices,
                const dal::array<std::int64_t>& row_offsets,
                std::int64_t row_count,
                std::int64_t column_count,
                sparse_indexing indexing) {
        const auto byte_data = detail::reinterpret_array_cast<byte_t>(data);
        get_impl().set_data_type(detail::make_data_type<Data>());
        get_impl().reset(byte_data, column_indices, row_offsets, row_count, column_count, indexing);
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Reset the internal data of the builder with the provided inputs.
    ///
    /// @tparam Data    The type of elements in the input values block.
    ///                 The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                 or :expr:`std::int32_t`.
    ///
    /// @param data           The array of values in the CSR layout.
    /// @param column_indices The array of column indices in the CSR layout.
    /// @param row_offsets    The array of row offsets in the CSR layout.
    /// @param row_count      The number of rows in the table to be built.
    /// @param column_count   The number of columns in the table to be built.
    /// @param indexing       The indexing scheme used to access data in the CSR layout.
    ///                       Should be :literal:`sparse_indexing::zero_based` or
    ///                       :literal:`sparse_indexing::one_based`.
    /// @param dependencies   Events indicating availability of the :literal:`data` for reading or writing.
    ///
    /// @result Reference to the updated CSR table builder.
    template <typename Data>
    auto& reset(const dal::array<Data>& data,
                const dal::array<std::int64_t>& column_indices,
                const dal::array<std::int64_t>& row_offsets,
                std::int64_t row_count,
                std::int64_t column_count,
                sparse_indexing indexing,
                const std::vector<sycl::event>& dependencies) {
        const auto byte_data = detail::reinterpret_array_cast<byte_t>(data);
        get_impl().set_data_type(detail::make_data_type<Data>());
        get_impl().reset(byte_data,
                         column_indices,
                         row_offsets,
                         row_count,
                         column_count,
                         indexing,
                         dependencies);
        return *this;
    }
#endif

private:
    csr_table_builder_iface& get_impl() {
        return cast_impl<csr_table_builder_iface>(*this);
    }
};

} // namespace v1

using v1::table_builder;
using v1::homogen_table_builder;
using v1::csr_table_builder;

} // namespace oneapi::dal::detail

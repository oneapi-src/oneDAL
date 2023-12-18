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

class ONEDAL_EXPORT table_builder {
    friend pimpl_accessor;

public:
    table build() const {
        return detail::make_private<table>(impl_->build());
    }

protected:
    explicit table_builder(table_builder_iface* impl) : impl_(impl) {}

private:
    pimpl<table_builder_iface> impl_;
};

/// Builds homogeneous table from the provided data
class ONEDAL_EXPORT homogen_table_builder : public table_builder {
public:
    homogen_table_builder();

    /// Builds homogeneous table
    homogen_table build() {
        return detail::make_private<homogen_table>(get_impl().build_homogen());
    }

    /// Reset the
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

    template <typename Data>
    auto& reset(const dal::array<Data>& data, std::int64_t row_count, std::int64_t column_count) {
        const auto byte_data = detail::reinterpret_array_cast<byte_t>(data);
        get_impl().set_data_type(detail::make_data_type<Data>());
        get_impl().reset(byte_data, row_count, column_count);
        return *this;
    }

    auto& set_data_type(data_type dt) {
        get_impl().set_data_type(dt);
        return *this;
    }

    auto& set_feature_type(feature_type ft) {
        get_impl().set_feature_type(ft);
        return *this;
    }

    auto& set_layout(data_layout layout) {
        get_impl().set_layout(layout);
        return *this;
    }

    auto& allocate(std::int64_t row_count, std::int64_t column_count) {
        get_impl().allocate(row_count, column_count);
        return *this;
    }

    template <typename Data>
    auto& copy_data(const Data* data, std::int64_t row_count, std::int64_t column_count) {
        get_impl().copy_data(data, row_count, column_count);
        return *this;
    }

    template <typename Data>
    auto& copy_data(const array<Data>& data) {
        get_impl().copy_data(reinterpret_array_cast<dal::byte_t>(data));
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    auto& allocate(const sycl::queue& queue,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        get_impl().allocate(detail::data_parallel_policy{ queue }, row_count, column_count, alloc);
        return *this;
    }

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

    csr_table build() {
        return detail::make_private<csr_table>(get_impl().build_csr());
    }

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

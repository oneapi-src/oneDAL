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

#include "oneapi/dal/table/backend/homogen_table_impl.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::backend {

class homogen_table_builder_impl {
public:
    homogen_table_builder_impl()
            : row_count_(0),
              column_count_(0),
              layout_(data_layout::row_major) {}

    void reset(homogen_table&& t) {
        auto& t_impl = detail::get_impl<detail::homogen_table_impl_iface>(t);
        auto& meta = t_impl.get_metadata();

        layout_ = t.get_data_layout();
        dtype_ = meta.get_data_type(0);

        std::int64_t data_size =
            detail::get_data_type_size(dtype_) * t_impl.get_row_count() * t_impl.get_column_count();

        // TODO: make data move without copying
        // now we are accepting const data pointer from table
        data_.reset(array<byte_t>(), reinterpret_cast<const byte_t*>(t_impl.get_data()), data_size);
        data_.need_mutable_data();
        row_count_ = t_impl.get_row_count();
        column_count_ = t_impl.get_column_count();
    }

    void reset(const array<byte_t>& data, std::int64_t row_count, std::int64_t column_count) {
        data_ = data;
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void set_data_type(data_type dt) {
        dtype_ = dt;
        data_.reset();
        row_count_ = 0;
        column_count_ = 0;
    }

    void set_feature_type(feature_type ft) {
        // TODO: not propagated to homogen_table_impl
    }

    void allocate(std::int64_t row_count, std::int64_t column_count) {
        data_.reset(row_count * column_count * detail::get_data_type_size(dtype_));
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void set_layout(data_layout layout) {
        layout_ = layout;
    }

    void copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) {
        data_.reset(row_count * column_count * detail::get_data_type_size(dtype_));
        detail::memcpy(detail::default_host_policy{},
                       data_.get_mutable_data(),
                       data,
                       data_.get_size());

        row_count_ = row_count;
        column_count_ = column_count;
    }

    homogen_table build() {
        homogen_table new_table{ homogen_table_impl{ column_count_, data_, dtype_, layout_ } };
        data_.reset();
        row_count_ = 0;
        column_count_ = 0;
        layout_ = data_layout::row_major;
        dtype_ = data_type::float32; // TODO: default data_type

        return new_table;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    void allocate(const sycl::queue& queue,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  sycl::usm::alloc kind) {
        data_.reset(queue, row_count * column_count * detail::get_data_type_size(dtype_), kind);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void copy_data(sycl::queue& queue,
                   const void* data,
                   std::int64_t row_count,
                   std::int64_t column_count) {
        data_.reset(queue,
                    row_count * column_count * detail::get_data_type_size(dtype_),
                    sycl::get_pointer_type(data_.get_data(), queue.get_context()));
        detail::memcpy(queue, data_.get_mutable_data(), data, data_.get_size());

        row_count_ = row_count;
        column_count_ = column_count;
    }
#endif

    // TODO: for better performance, push_*() methods need to be moved
    // from table implementation to builder.
    // pull_*() methods can be generalized between table and builder
    template <typename T>
    void pull_rows(array<T>& a, const range& r) const {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.pull_rows(a, r);
    }

    template <typename T>
    void push_rows(const array<T>& a, const range& r) {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.push_rows(a, r);
    }

    template <typename T>
    void pull_column(array<T>& a, std::int64_t idx, const range& r) const {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.pull_column(a, idx, r);
    }

    template <typename T>
    void push_column(const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.push_column(a, idx, r);
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename T>
    void pull_rows(sycl::queue& q,
                   array<T>& a,
                   const range& r,
                   const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.pull_rows(a, r);
    }

    template <typename T>
    void push_rows(sycl::queue& q, const array<T>& a, const range& r) {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.push_rows(a, r);
    }

    template <typename T>
    void pull_column(sycl::queue& q,
                     array<T>& a,
                     std::int64_t idx,
                     const range& r,
                     const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.pull_column(a, idx, r);
    }

    template <typename T>
    void push_column(sycl::queue& q, const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ column_count_, data_, dtype_, layout_ };
        impl.push_column(a, idx, r);
    }
#endif

private:
    array<byte_t> data_;
    int64_t row_count_;
    int64_t column_count_;
    data_layout layout_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend

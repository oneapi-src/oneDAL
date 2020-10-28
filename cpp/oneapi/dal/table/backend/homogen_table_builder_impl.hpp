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
    homogen_table_builder_impl() {
        reset();
    }

    void reset() {
        data_.reset();
        row_count_ = 0;
        column_count_ = 0;
        layout_ = data_layout::row_major;
        dtype_ = data_type::float32;
    }

    void reset(homogen_table&& t) {
        if (t.has_data()) {
            const auto& meta = t.get_metadata();
            const std::int64_t data_size =
                get_data_size(t.get_row_count(), t.get_column_count(), meta.get_data_type(0));

            // TODO: make data move without copying
            // now we are accepting const data pointer from table
            data_.reset(reinterpret_cast<const byte_t*>(t.get_data()),
                        data_size,
                        empty_delete<const byte_t>());
            data_.need_mutable_data();

            layout_ = t.get_data_layout();
            dtype_ = meta.get_data_type(0);
            row_count_ = t.get_row_count();
            column_count_ = t.get_column_count();
        }
        else {
            reset();
        }
    }

    void reset(const array<byte_t>& data, std::int64_t row_count, std::int64_t column_count) {
        if (get_data_size(row_count, column_count, dtype_) != data.get_count()) {
            throw dal::range_error("invalid data size");
        }

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
        throw dal::unimplemented("set_feature_type is not implemented");
    }

    void allocate(std::int64_t row_count, std::int64_t column_count) {
        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        if (data_size <= 0) {
            throw dal::domain_error("data count to allocate is not positive");
        }

        data_.reset(data_size);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void set_layout(data_layout layout) {
        layout_ = layout;
    }

    void copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) {
        allocate(row_count, column_count);
        detail::memcpy(detail::default_host_policy{}, data_.get_mutable_data(), data, data_.get_size());
    }

    homogen_table build() {
        homogen_table new_table{
            homogen_table_impl{ row_count_, column_count_, data_, dtype_, layout_ }
        };

        reset();
        return new_table;
    }

#ifdef ONEDAL_DATA_PARALLEL
    void allocate(const sycl::queue& queue,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  sycl::usm::alloc kind) {
        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        if (data_size <= 0) {
            throw dal::domain_error("data count to allocate is not positive");
        }

        data_.reset(queue, data_size, kind);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void copy_data(sycl::queue& queue,
                   const void* data,
                   std::int64_t row_count,
                   std::int64_t column_count) {
        const auto kind = sycl::get_pointer_type(data_.get_data(), queue.get_context());
        ONEDAL_ASSERT(kind != sycl::usm::alloc::unknown);

        allocate(queue, row_count, column_count, kind);
        detail::memcpy(queue, data_.get_mutable_data(), data, data_.get_size());
    }
#endif

    // TODO: for better performance, push_*() methods need to be moved
    // from table implementation to builder.
    // pull_*() methods can be generalized between table and builder
    template <typename T>
    void pull_rows(array<T>& a, const range& r) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.pull_rows(a, r);
    }

    template <typename T>
    void push_rows(const array<T>& a, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.push_rows(a, r);
    }

    template <typename T>
    void pull_column(array<T>& a, std::int64_t idx, const range& r) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.pull_column(a, idx, r);
    }

    template <typename T>
    void push_column(const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.push_column(a, idx, r);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows(sycl::queue& q,
                   array<T>& a,
                   const range& r,
                   const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.pull_rows(q, a, r, kind);
    }

    template <typename T>
    void push_rows(sycl::queue& q, const array<T>& a, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.push_rows(q, a, r);
    }

    template <typename T>
    void pull_column(sycl::queue& q,
                     array<T>& a,
                     std::int64_t idx,
                     const range& r,
                     const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.pull_column(q, a, idx, r, kind);
    }

    template <typename T>
    void push_column(sycl::queue& q, const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, dtype_, layout_ };
        impl.push_column(q, a, idx, r);
    }
#endif

private:
    static std::int64_t get_data_size(std::int64_t row_count,
                                      std::int64_t column_count,
                                      data_type dtype) {
        detail::check_mul_overflow(row_count, column_count);
        const std::int64_t element_count = row_count * column_count;
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        return element_count * dtype_size;
    }

private:
    array<byte_t> data_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    data_layout layout_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend

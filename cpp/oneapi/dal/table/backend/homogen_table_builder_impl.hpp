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
                        detail::empty_delete<const byte_t>());
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
            throw dal::range_error(dal::detail::error_messages::invalid_data_block_size());
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

    void set_feature_type(feature_type ft, std::int64_t idx) {
        if (column_count_ == 0) {
            throw dal::domain_error{ "column_count should be > 0 " };
        }
        ONEDAL_ASSERT(idx >= 0);
        ONEDAL_ASSERT(idx < column_count_);

        allocate_ftypes_if_need();
        feature_type* ftypes_ptr = ftypes_.need_mutable_data().get_mutable_data();
        ftypes_ptr[idx] = ft;
    }

    void set_layout(data_layout layout) {
        layout_ = layout;
    }

    void allocate(std::int64_t row_count, std::int64_t column_count) {
        if (row_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::cc_leq_zero());
        }

        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        data_.reset(data_size);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) {
        check_copy_data_preconditions(row_count, column_count);
        detail::memcpy(detail::default_host_policy{},
                       data_.get_mutable_data(),
                       data,
                       data_.get_size());
    }

    homogen_table build() {
        homogen_table new_table{
            homogen_table_impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ }
        };

        reset();
        return new_table;
    }

#ifdef ONEDAL_DATA_PARALLEL
    void allocate(const sycl::queue& queue,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  sycl::usm::alloc kind) {
        if (row_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::cc_leq_zero());
        }

        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        data_.reset(queue, data_size, kind);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void copy_data(sycl::queue& queue,
                   const void* data,
                   std::int64_t row_count,
                   std::int64_t column_count) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data_.get_data(), queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        check_copy_data_preconditions(row_count, column_count);
        detail::memcpy(queue, data_.get_mutable_data(), data, data_.get_size());
    }
#endif

    // TODO: for better performance, push_*() methods need to be moved
    // from table implementation to builder.
    // pull_*() methods can be generalized between table and builder
    template <typename T>
    void pull_rows(array<T>& a, const range& r) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.pull_rows(a, r);
    }

    template <typename T>
    void push_rows(const array<T>& a, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.push_rows(a, r);
    }

    template <typename T>
    void pull_column(array<T>& a, std::int64_t idx, const range& r) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.pull_column(a, idx, r);
    }

    template <typename T>
    void push_column(const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.push_column(a, idx, r);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows(sycl::queue& q,
                   array<T>& a,
                   const range& r,
                   const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.pull_rows(q, a, r, kind);
    }

    template <typename T>
    void push_rows(sycl::queue& q, const array<T>& a, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.push_rows(q, a, r);
    }

    template <typename T>
    void pull_column(sycl::queue& q,
                     array<T>& a,
                     std::int64_t idx,
                     const range& r,
                     const sycl::usm::alloc& kind) const {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
        impl.pull_column(q, a, idx, r, kind);
    }

    template <typename T>
    void push_column(sycl::queue& q, const array<T>& a, std::int64_t idx, const range& r) {
        homogen_table_impl impl{ row_count_, column_count_, data_, ftypes_, dtype_, layout_ };
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

    void check_copy_data_preconditions(std::int64_t row_count, std::int64_t column_count) {
        const std::int64_t reqired_size = get_data_size(row_count, column_count, dtype_);
        const std::int64_t allocated_size = get_data_size(row_count_, column_count_, dtype_);
        if (allocated_size < reqired_size) {
            throw dal::invalid_argument{
                dal::detail::error_messages::allocated_memory_size_is_not_enough_to_copy_data()
            };
        }
    }

    void allocate_ftypes_if_need() {
        ONEDAL_ASSERT(column_count_ != 0);
        if (ftypes_.get_count() == 0) {
            auto default_ftype =
                detail::is_floating_point(dtype_) ? feature_type::ratio : feature_type::ordinal;

            ftypes_ = array<feature_type>::full(column_count_, default_ftype);
        }
        ONEDAL_ASSERT(column_count_ == ftypes_.get_count());
    }

    array<byte_t> data_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    data_layout layout_;
    data_type dtype_;
    array<feature_type> ftypes_;
};

} // namespace oneapi::dal::backend

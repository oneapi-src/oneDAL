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
#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

class csr_table_builder_impl
        : public detail::csr_table_builder_template<csr_table_builder_impl> {
public:
    csr_table_builder_impl() {
        reset();
    }

    void reset() {
        data_.reset();
        column_indices_.reset();
        row_indices_.reset();
        column_count_ = 0;
        layout_ = data_layout::row_major;
        dtype_ = data_type::float32;
    }

    void reset(const dal::array<byte_t>& data,
               const dal::array<std::int64_t>& column_indices,
               const dal::array<std::int64_t>& row_indices,
               std::int64_t column_count) override {
        if (column_indices.get_count() != data.get_count()) {
            throw dal::range_error(dal::detail::error_messages::invalid_data_block_size());
        }

        data_ = data;
        column_indices_ = column_indices;
        row_indices_ = row_indices;
        column_count_ = column_count;
    }

    void set_data_type(data_type dt) override {
        dtype_ = dt;
        data_.reset();
        column_indices_.reset();
        row_indices_.reset();
        column_count_ = 0;
    }

    void set_feature_type(feature_type ft) override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    virtual void allocate(std::int64_t element_count, //
                          std::int64_t row_count) override {
        if (element_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::ec_leq_zero());
        }

        if (row_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::rc_leq_zero());
        }

        const std::int64_t data_size = get_data_size(element_count, dtype_);
        data_.reset(data_size);
        column_indices_.reset(element_count);
        row_indices_.reset(row_count + 1);
        column_count_ = column_count;
    }

    void set_layout(data_layout layout) override {
        layout_ = layout;
    }

    void copy_data(const void* data,
                   const int64_t* column_indices,
                   const int64_t* row_indices,
                   std::int64_t element_count,
                   std::int64_t row_count) override {
        check_copy_data_preconditions(row_count, column_count);

        __ONEDAL_IF_QUEUE__(data_.get_queue(), {
            auto this_q = data_.get_queue().value();
            ONEDAL_ASSERT(is_known_usm(data_));
            ONEDAL_ASSERT(is_known_usm(column_indices_));
            ONEDAL_ASSERT(is_known_usm(row_indices_));
            detail::memcpy_host2usm(this_q, data_.get_mutable_data(), data, data_.get_size());
            detail::memcpy_host2usm(this_q, column_indices_.get_mutable_data(), column_indices,
                column_indices_.get_size());
            detail::memcpy_host2usm(this_q, row_indices_.get_mutable_data(), row_indices,
                row_indices_.get_size());
        });

        __ONEDAL_IF_NO_QUEUE__(data_.get_queue(), {
            detail::memcpy(detail::default_host_policy{},
                           data_.get_mutable_data(),
                           data,
                           data_.get_size());
            detail::memcpy(detail::default_host_policy{},
                           column_indices_.get_mutable_data(),
                           column_indices_,
                           column_indices_.get_size());
            detail::memcpy(detail::default_host_policy{},
                           row_indices_.get_mutable_data(),
                           row_indices_,
                           row_indices_.get_size());
        });
    }

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {
        constexpr bool preserve_mutability = true;
        csr_pull_rows(policy, get_info(), data_, column_indices_, row_indices_,
                      block, rows, alloc_kind::host, preserve_mutability);
    }

    template <typename T>
    void pull_csr_block_template(const default_host_policy& policy,
                        csr_block<T>& block,
                        const csr_indexing& indexing,
                        const range& rows) const {
        constexpr bool preserve_mutability = true;

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count()), indexing };

        csr_pull_block(policy, get_info(), block_info, data_, column_indices_, row_indices_,
                       block, alloc_kind::host, preserve_mutability);

    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {
        constexpr bool preserve_mutability = true;
        csr_pull_rows(policy, get_info(), data_, column_indices_, row_indices_,
                      block, rows, alloc_kind_from_sycl(alloc), preserve_mutability);
    }

    template <typename T>
    void pull_csr_block_template(const detail::data_parallel_policy& policy,
                        csr_block<T>& block,
                        const csr_indexing& indexing,
                        const range& row_range) const {
        constexpr bool preserve_mutability = true;

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count()), indexing };

        csr_pull_block(policy, get_info(), block_info, data_, column_indices_, row_indices_,
                       block, alloc_kind_from_sycl(alloc), preserve_mutability);

    }
#endif

private:
    void check_copy_data_preconditions(std::int64_t element_count, std::int64_t row_count) {
        const std::int64_t reqired_data_size = get_data_size(element_count, dtype_);
        const std::int64_t allocated_data_size = data_.get_size();
        if (allocated_size < reqired_size) {
            using msg = dal::detail::error_messages;
            throw dal::invalid_argument{ msg::allocated_memory_size_is_not_enough_to_copy_data() };
        }
        if (row_count() < row_count) {
            using msg = dal::detail::error_messages;
            throw dal::invalid_argument{ msg::allocated_memory_size_is_not_enough_to_copy_row_indices() };
        }
    }

    static std::int64_t get_data_size(std::int64_t element_count,
                                      data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        detail::check_mul_overflow(element_count, dtype_size);
        return element_count * dtype_size;
    }

    inline std::int64_t row_count() const {
        std::int64_t row_indices_count = row_indices_.get_count();
        return (row_indices_count ? row_indices_count - 1 : 0);
    }

    array<byte_t> data_;
    array<std::int64_t> column_indices_;
    array<std::int64_t> row_indices_;
    std::int64_t column_count_;
    data_layout layout_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend
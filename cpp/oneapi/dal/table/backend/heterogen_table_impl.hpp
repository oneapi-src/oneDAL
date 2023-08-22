/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <iostream>
#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/heterogen_kernels.hpp"

#include "oneapi/dal/table/detail/table_kinds.hpp"
#include "oneapi/dal/table/detail/metadata_utils.hpp"

#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::backend {

class heterogen_table_impl : public detail::heterogen_table_template<heterogen_table_impl>,
                             public ONEDAL_SERIALIZABLE(heterogen_table_id) {
public:
    heterogen_table_impl() {}

    heterogen_table_impl(std::int64_t column_count) {
        auto empty = detail::chunked_array_base{};
        data_ = array<detail::chunked_array_base>::full(column_count, empty);
    }

    heterogen_table_impl(const table_metadata& meta)
            : heterogen_table_impl{ meta.get_feature_count() } {
        meta_ = table_metadata{ meta };
    }

    std::int64_t get_row_count() const override {
        ONEDAL_ASSERT(validate());
        if (get_column_count() == 0l)
            return 0l;
        auto dt = get_metadata().get_data_type(0l);
        return detail::get_element_count(dt, data_[0l]);
    }

    std::int64_t get_column_count() const override {
        return get_metadata().get_feature_count();
    }

    const table_metadata& get_metadata() const override {
        return meta_;
    }

    std::int64_t get_kind() const override {
        return detail::get_heterogen_table_kind();
    }

    data_layout get_data_layout() const override {
        return data_layout::column_major;
    }

    void serialize(detail::output_archive& ar) const override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    void deserialize(detail::input_archive& ar) override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    // virtual void set_column(std::int64_t, data_type, detail::chunked_array_base) = 0;
    void set_column(std::int64_t column, data_type dt, detail::chunked_array_base arr) override {
        ONEDAL_ASSERT(column < get_column_count());
        auto* const ptr = data_.get_mutable_data();
        *(ptr + column) = std::move(arr);
    }

    //virtual const detail::chunked_array_base& get_column(std::int64_t) const = 0;
    const detail::chunked_array_base& get_column(std::int64_t column) const override {
        ONEDAL_ASSERT(column < get_column_count());
        const auto* const ptr = data_.get_data();
        return *(ptr + column);
    }

    // virtual detail::chunked_array_base& get_column(std::int64_t) = 0;
    detail::chunked_array_base& get_column(std::int64_t column) override {
        ONEDAL_ASSERT(column < get_column_count());
        auto* const ptr = data_.get_mutable_data();
        return *(ptr + column);
    }

    bool validate() const {
        const auto col_count = get_column_count();

        if (col_count <= std::int64_t{ 1l }) {
            return true;
        }

        const auto dt = get_metadata().get_data_type(0l);
        const auto row_count = detail::get_element_count(dt, data_[0l]);

        for (std::int64_t c = 1l; c < col_count; ++c) {
            const auto dt_col = get_metadata().get_data_type(c);
            const auto count = detail::get_element_count(dt_col, data_[c]);

            if (count != row_count) {
                return false;
            }
        }

        return true;
    }

    detail::access_iface_host& get_access_iface_host() const override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#ifdef ONEDAL_DATA_PARALLEL
    detail::access_iface_dpc& get_access_iface_dpc() const override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {
        heterogen_pull_rows(policy, meta_, data_, block, rows, alloc_kind::host);
    }

    template <typename T>
    void pull_column_template(const detail::default_host_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows) const {
        heterogen_pull_column(policy, meta_, data_, block, column_index, rows, alloc_kind::host);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {
        const alloc_kind req_alloc = alloc_kind_from_sycl(alloc);
        heterogen_pull_rows(policy, meta_, data_, block, rows, req_alloc);
    }

    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {
        const alloc_kind req_alloc = alloc_kind_from_sycl(alloc);
        heterogen_pull_column(policy, meta_, data_, block, column_index, rows, req_alloc);
    }
#endif

private:
    table_metadata meta_;
    dal::array<detail::chunked_array_base> data_;
};

} // namespace oneapi::dal::backend

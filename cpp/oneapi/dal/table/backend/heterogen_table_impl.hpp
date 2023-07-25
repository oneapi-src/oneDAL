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
        data_ = dal::array<detail::chunked_array_base>::empty(column_count);
    }

    std::int64_t get_row_count() const override {
        ONEDAL_ASSERT(validate());
        const auto dt = get_metadata().get_data_type(0l);
        const auto elem = detail::get_data_type_size(dt);
        const auto size = get_column(0l).get_size_in_bytes();

        ONEDAL_ASSERT(size % elem == 0l);
        return size / elem;
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

    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {

    }

    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {

    }
#endif

private:
    table_metadata meta_;
    dal::array<detail::chunked_array_base> data_;
};

} // namespace oneapi::dal::backend

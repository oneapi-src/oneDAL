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

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/backend/heterogen_kernels.hpp"
#include "oneapi/dal/table/backend/heterogen_table_impl.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

/*class heterogen_table_builder_impl
        : public detail::heterogen_table_builder_template<heterogen_table_builder_impl> {
public:
    homogen_table_builder_impl() {
        reset();
    }

    void reset() {
        data_.reset();
    }

    void set_feature_type(feature_type ft) override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    detail::heterogen_table_iface* build_heterogen() override {
        auto new_table =
            new heterogen_table_impl{ row_count_, column_count_, data_, dtype_, layout_ };
        reset();
        return new_table;
    }

    detail::heterogen_table_iface* build() override {
        return build_heterogen();
    }

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void pull_column_template(const detail::default_host_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows) const {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void push_rows_template(const detail::default_host_policy& policy,
                            const array<T>& block,
                            const range& rows) {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void push_column_template(const detail::default_host_policy& policy,
                              const array<T>& block,
                              std::int64_t column_index,
                              const range& rows) {
       throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void push_rows_template(const detail::data_parallel_policy& policy,
                            const array<T>& block,
                            const range& rows) {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    void push_column_template(const detail::data_parallel_policy& policy,
                              const array<T>& block,
                              std::int64_t column_index,
                              const range& rows) {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif

private:
    table_metadata metadata_
    dal::array<detail::chunked_array_base> data_;
};*/

} // namespace oneapi::dal::backend

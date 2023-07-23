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

/*class heterogen_table_impl : public detail::heterogen_table_template<heterogen_table_impl>,
                             public ONEDAL_SERIALIZABLE(heterogen_table_id) {
public:
    heterogen_table_impl() : col_count_{ 0l } {}

    heterogen_table_impl(std::int64_t column_count) : col_count_{ column_count } {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }


    std::int64_t get_column_count() const override {
        return col_count_;
    }

    std::int64_t get_row_count() const override {
        return row_count_;
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


    bool validate() const {
        return true;
    }

private:

    table_metadata meta_;
    std::int64_t col_count_;
    std::int64_t row_count_;
    //dal::array<chunked_array_base>
};*/

} // namespace oneapi::dal::backend

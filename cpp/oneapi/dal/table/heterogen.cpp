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

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/detail/table_kinds.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"
#include "oneapi/dal/table/backend/heterogen_table_impl.hpp"

namespace oneapi::dal {

static std::shared_ptr<detail::heterogen_table_iface> get_heterogen_iface(const table& other) {
    if (const auto heterogen_iface = detail::get_heterogen_table_iface(other)) {
        return heterogen_iface;
    }
    return std::make_shared<backend::heterogen_table_impl>();
}

std::int64_t heterogen_table::kind() {
    return detail::get_heterogen_table_kind();
}

heterogen_table::heterogen_table() : heterogen_table(new backend::heterogen_table_impl{}) {}

heterogen_table::heterogen_table(const table& other)
        : heterogen_table(get_heterogen_iface(other)) {}

heterogen_table heterogen_table::empty(const table_metadata& meta) {
    auto* const impl = new backend::heterogen_table_impl(meta);
    return heterogen_table{ impl };
}

void heterogen_table::set_column_impl(std::int64_t column,
                                      data_type dt,
                                      detail::chunked_array_base arr) {
    auto& impl = detail::cast_impl<detail::heterogen_table_iface>(*this);
    ONEDAL_ASSERT(dt == this->get_metadata().get_data_type(column));
    impl.set_column(column, dt, std::move(arr));
}

std::pair<data_type, detail::chunked_array_base> heterogen_table::get_column_impl(
    std::int64_t column) const {
    data_type dtype = this->get_metadata().get_data_type(column);
    auto array = detail::cast_impl<const detail::heterogen_table_iface>(*this).get_column(column);
    return std::pair<data_type, detail::chunked_array_base>(std::move(dtype), std::move(array));
}

} // namespace oneapi::dal

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

#include "oneapi/dal/table/backend/interop/table_conversion_common.hpp"
#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/common.hpp"

#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::backend::interop {

heterogen_table convert_to_heterogen(const soa_table_ptr_t& t) {
    return convert_to_heterogen(*t);
}

heterogen_table convert_to_heterogen(soa_table_t& t) {
    constexpr std::size_t zero = 0;
    using dal::detail::integral_cast;
    using dal::detail::check_mul_overflow;
    using dal::detail::check_sum_overflow;

    const auto raw_row_count = t.getNumberOfRows();
    const auto raw_col_count = t.getNumberOfColumns();

    const auto row_count = integral_cast<std::int64_t>(raw_row_count);
    const auto col_count = integral_cast<std::int64_t>(raw_col_count);

    const auto daal_meta = t.getDictionarySharedPtr();
    const table_metadata dal_meta = convert(daal_meta);

    auto table = heterogen_table::empty(dal_meta);
    for (std::int64_t col = 0l; col < col_count; ++col) {
        const auto dal_dtype = dal_meta.get_data_type(col);

        dispatch_by_table_type(
            [&](auto type_tag) -> void {
                using type_t = std::decay_t<decltype(type_tag)>;

                daal_dm::BlockDescriptor<type_t> column_block;
                auto s = t.getBlockOfColumnValues(col,
                                                  zero,
                                                  raw_col_count,
                                                  daal_dm::ReadWriteMode::readOnly,
                                                  column_block);
                interop::status_to_exception(s);

                auto shared_ptr = column_block.getBlockSharedPtr();
                const type_t* const raw_ptr = shared_ptr.get();

                auto deleter = [shared_ptr, t](const type_t*) {};
                dal::array<type_t> column(raw_ptr, row_count, deleter);

                table.set_column(col, std::move(column));
            },
            [](auto dummy) -> void {
                using msg = detail::error_messages;
                auto str = msg::unsupported_table_conversion();
                throw dal::unimplemented(str);
            },
            dal_dtype);
    }

    return table;
}

} // namespace oneapi::dal::backend::interop

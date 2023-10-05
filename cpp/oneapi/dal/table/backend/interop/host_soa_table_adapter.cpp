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

#include <daal/include/data_management/data/soa_numeric_table.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_homogen_to_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_heterogen_to_soa_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

template <typename DefaultType>
soa_table_ptr_t convert_to_soa(const table& t, DefaultType type_tag) {
    soa_table_ptr_t result;
    const auto kind = t.get_kind();

    if (kind == heterogen_table::kind()) {
        const auto& raw = reinterpret_cast<const homogen_table&>(t);
        auto ptr = host_soa_table_adapter::create<DefaultType>(raw);
        result = soa_table_ptr_t{ ptr };
    }
    else if (kind == homogen_table::kind()) {
        const auto& raw = reinterpret_cast<const heterogen_table&>(t);
        auto ptr = host_heterogen_table_adapter::create(raw);
        result = soa_table_ptr_t{ ptr };
    }
    else {
        using msg = dal::detail::error_messages;
        throw invalid_argument(msg::unsupported_table_conversion());
    }

    return result;
}

#define INSTANTIATE(TYPE) \
    template ONEDAL_EXPORT soa_table_ptr_t convert_to_soa<TYPE>(const table&, TYPE);

INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::interop
